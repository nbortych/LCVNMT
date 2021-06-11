# coding: utf-8
"""
Training module
"""

import argparse
import time
import shutil
from typing import List
import logging
import os
import sys
import collections
import pathlib
import numpy as np
import pickle
import copy
import operator

import torch
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Dataset

from joeynmt.model import build_model
from joeynmt.batch import Batch
from joeynmt.helpers import log_data_info, load_config, log_cfg, \
    store_attention_plots, load_checkpoint, make_model_dir, \
    make_logger, set_seed, symlink_update, latest_checkpoint_update, \
    ConfigurationError
from joeynmt.model import Model, _DataParallel, _DistributedDataParallel
from joeynmt.prediction import validate_on_data
from joeynmt.loss import XentLoss
from joeynmt.data import load_data, make_dataloader, BatchSamplerSimilarLength, \
    DistributedBatchSamplerSimilarLength
from joeynmt.builders import build_optimizer, build_scheduler, \
    build_gradient_clipper
from joeynmt.prediction import test
from joeynmt.vocabulary import build_vocab

# for debug purposes
# torch.autograd.set_detect_anomaly(True)

# for fp16 training
try:
    from apex import amp

    amp.register_half_function(torch, "einsum")
except ImportError as no_apex:
    # error handling in TrainManager object construction
    pass

try:
    import wandb
except ImportError as no_wandb:
    pass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# pylint: disable=too-many-instance-attributes
class TrainManager:
    """ Manages training loop, validations, learning rate scheduling
    and early stopping."""

    def __init__(self, model: Model, config: dict,
                 batch_class: Batch = Batch) -> None:
        """
        Creates a new TrainManager for a model, specified as in configuration.

        :param model: torch module defining the model
        :param config: dictionary containing the training configurations
        :param batch_class: batch class to encapsulate the torch class
        """
        self.config = copy.deepcopy(config)
        train_config = config["training"]
        self.train_config = train_config.copy()
        self.batch_class = batch_class

        # files for logging and storing
        self.model_dir = train_config["model_dir"]
        assert os.path.exists(self.model_dir)

        # are we really training or just debugging
        self.small_test_run = train_config.get('small_test_run', False)
        logger.info(f"Small test run? {self.small_test_run}")

        # logging
        self.logging_freq = train_config.get("logging_freq", 100)
        self.valid_report_file = "{}/validations.txt".format(self.model_dir)
        self.tb_writer = None  # set inside the training due to ddp
        self.save_latest_checkpoint = train_config.get("save_latest_ckpt", True)

        # using wandb to log hyperparams + log results
        self.use_wandb = train_config.get("wandb", False)

        # model
        self.model = model
        self._log_parameters_list()
        # log gradients of the model to wandb
        # if self.use_wandb:
        #     wandb.watch(self.model, log_freq=1000)

        # generation
        self.max_output_length = train_config.get("max_output_length", None)

        # objective
        self.utility_regularising = train_config.get("utility_regularising", False)
        self._utility_alpha = train_config.get("utility_alpha", 1)
        self.label_smoothing = train_config.get("label_smoothing", 0.0)
        self._num_samples = train_config.get("num_samples", 10)
        self._mean_baseline = train_config.get("mean_baseline", False)
        self._vimco_baseline = train_config.get("vimco_baseline", False)

        # self.model._utility_alpha = self.utility_alpha
        # self.model._num_samples = self.num_samples
        self.model.loss_function = XentLoss(pad_index=self.model.pad_index,
                                            smoothing=self.label_smoothing,
                                            utility_alpha=self._utility_alpha,
                                            num_samples=self._num_samples,
                                            max_output_length=self.max_output_length,
                                            mean_baseline=self._mean_baseline,
                                            vimco_baseline=self._vimco_baseline)
        self.normalization = train_config.get("normalization", "batch")
        if self.normalization not in ["batch", "tokens", "none"]:
            raise ConfigurationError("Invalid normalization option."
                                     "Valid options: "
                                     "'batch', 'tokens', 'none'.")

        # optimization
        self.learning_rate_min = train_config.get("learning_rate_min", 1.0e-8)

        self.clip_grad_fun = build_gradient_clipper(config=train_config)
        self.optimizer = build_optimizer(config=train_config,
                                         parameters=model.parameters())

        # validation & early stopping
        self.track_mbr = train_config.get("track_mbr", False)
        self.utility_type = train_config.get("utility", "editdistance")

        self.validation_freq = train_config.get("validation_freq", 1000)
        self.log_valid_sents = train_config.get("print_valid_sents", [0, 1, 2])
        self.ckpt_queue = collections.deque(
            maxlen=train_config.get("keep_last_ckpts", 5))
        self.eval_metric = train_config.get("eval_metric", "bleu")
        if self.eval_metric not in [
            'bleu', 'chrf', 'token_accuracy', 'sequence_accuracy', ''
        ]:
            raise ConfigurationError("Invalid setting for 'eval_metric', "
                                     "valid options: 'bleu', 'chrf', "
                                     "'token_accuracy', 'sequence_accuracy'.")
        self.early_stopping_metric = train_config.get("early_stopping_metric",
                                                      "eval_metric")

        # early_stopping_metric decides on how to find the early stopping point:
        # ckpts are written when there's a new high/low score for this metric.
        # If we schedule after BLEU/chrf/accuracy, we want to maximize the
        # score, else we want to minimize it.
        if self.early_stopping_metric in ["ppl", "loss"]:
            self.minimize_metric = True
        elif self.early_stopping_metric == "eval_metric":
            if self.eval_metric in [
                "bleu", "chrf", "token_accuracy", "sequence_accuracy"
            ]:
                self.minimize_metric = False
            # eval metric that has to get minimized (not yet implemented)
            else:
                self.minimize_metric = True
        elif self.early_stopping_metric == "utility":
            if self.utility_type == "beer":
                self.minimize_metric = False
            elif self.utility_type == "editdistance":
                self.minimize_metric = True
            else:
                raise ConfigurationError(
                    f"utility_type must be in one of [beer, editdistance]. It is currently {self.utility_type}")
        else:
            raise ConfigurationError(
                "Invalid setting for 'early_stopping_metric', "
                "valid options: 'loss', 'ppl', 'eval_metric', 'utility'.")

        # eval options
        test_config = config["testing"]
        self.bpe_type = test_config.get("bpe_type", "subword-nmt")
        self.sacrebleu = {"remove_whitespace": True, "tokenize": "13a"}
        if "sacrebleu" in config["testing"].keys():
            self.sacrebleu["remove_whitespace"] = test_config["sacrebleu"] \
                .get("remove_whitespace", True)
            self.sacrebleu["tokenize"] = test_config["sacrebleu"] \
                .get("tokenize", "13a")

        self.sample = test_config.get("sample", False)
        # learning rate scheduling
        self.scheduler, self.scheduler_step_at = build_scheduler(
            config=train_config,
            scheduler_mode="min" if self.minimize_metric else "max",
            optimizer=self.optimizer,
            hidden_size=config["model"]["encoder"]["hidden_size"])

        # data & batch handling
        self.level = config["data"]["level"]
        if self.level not in ["word", "bpe", "char"]:
            raise ConfigurationError("Invalid segmentation level. "
                                     "Valid options: 'word', 'bpe', 'char'.")
        self.shuffle = train_config.get("shuffle", True)
        self.epochs = train_config["epochs"]
        self.epoch_no = 0
        self.batch_size = train_config["batch_size"]
        self.early_stopping = train_config.get("early_stopping", True)
        self.early_stopping_patience = train_config.get("early_stopping_patience", 10)
        # do not stop training untill we early stop
        if self.early_stopping:
            self.epochs == sys.maxsize * 5
        # Placeholder so that we can use the train_iter in other functions.
        self.dataloader = None
        self.train_iter_state = None
        # per-device batch_size = self.batch_size // self.n_gpu
        self.batch_type = train_config.get("batch_type", "sentence")
        self.eval_batch_size = train_config.get("eval_batch_size",
                                                self.batch_size)
        # per-device eval_batch_size = self.eval_batch_size // self.n_gpu
        self.eval_batch_type = train_config.get("eval_batch_type",
                                                self.batch_type)

        self.batch_multiplier = train_config.get("batch_multiplier", 1)

        # CPU / GPU
        self.use_cuda = train_config["use_cuda"] and torch.cuda.is_available()
        self.n_gpu = torch.cuda.device_count() if self.use_cuda else 0
        logger.info(f"NUMBER OF GPUS IS {self.n_gpu}{'!' * 100}")
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        self.model.device = self.device
        # DDP
        # if there is no cuda available, do not use DDP
        self.ddp = train_config.get("ddp", False) if self.use_cuda else False
        self.model.ddp = self.ddp
        self.rank = None
        self.num_nodes = train_config.get("num_nodes", 1)
        self.node_nr = train_config.get("node_nr", 0)
        self.world_size = self.num_nodes * self.n_gpu
        self.distributed_batch_sampler = train_config.get("distributed_batch_sampler", False)
        self.child_conn = None  # set by child process in case of ddp
        # if self.use_cuda:
        #     torch.cuda.empty_cache()
        if self.use_cuda and not self.ddp:
            # do not put on device in case of ddp, since it will be put to specific devices
            self.model.to(self.device)

        # if not ddp, initialise wandb here
        if not self.ddp and self.use_wandb:
            self.init_wandb(config)
            wandb.watch(model)
        # fp16
        self.fp16 = train_config.get("fp16", False)
        if self.fp16:
            if 'apex' not in sys.modules:
                raise ImportError("Please install apex from "
                                  "https://www.github.com/nvidia/apex "
                                  "to use fp16 training.") from no_apex
            self.model, self.optimizer = amp.initialize(self.model,
                                                        self.optimizer,
                                                        opt_level='O1')
            # opt level: one of {"O0", "O1", "O2", "O3"}
            # see https://nvidia.github.io/apex/amp.html#opt-levels

        # initialize training statistics
        self.stats = self.TrainStatistics(
            steps=0,
            stop=False,
            total_tokens=0,
            best_ckpt_iter=0,
            best_ckpt_score=np.inf if self.minimize_metric else -np.inf,
            minimize_metric=self.minimize_metric,
            early_stopping_patience=self.early_stopping_patience)

        # model parameters
        if "load_model" in train_config.keys() and not self.ddp:
            self.init_from_checkpoint(
                train_config["load_model"],
                reset_best_ckpt=train_config.get("reset_best_ckpt", False),
                reset_scheduler=train_config.get("reset_scheduler", False),
                reset_optimizer=train_config.get("reset_optimizer", False),
                reset_iter_state=train_config.get("reset_iter_state", False))

        # multi-gpu training (should be after apex fp16 initialization)
        if self.n_gpu > 1 and not self.ddp:
            self.model = _DataParallel(self.model)
        self.train_sampler = None
        self.batch_sampler = None

    @staticmethod
    def init_wandb(config):
        if 'wandb' not in sys.modules:
            raise ImportError("Please install wandb to log with it ") from no_wandb
        with open('wandb_api_key.txt', 'r') as f:
            api_key = f.readline()
        wandb.login(key=api_key)
        wandb.init(project="lcv-nmt", config=config)

    def _save_checkpoint(self, new_best: bool = True) -> None:
        """
        Save the model's current parameters and the training state to a
        checkpoint.

        The training state contains the total number of training steps,
        the total number of training tokens,
        the best checkpoint score and iteration so far,
        and optimizer and scheduler states.

        :param new_best: This boolean signals which symlink we will use for the
          new checkpoint. If it is true, we update best.ckpt, else latest.ckpt.
        """
        model_path = os.path.join(self.model_dir,
                                  "{}.ckpt".format(self.stats.steps))
        model_state_dict = self.model.module.state_dict() \
            if isinstance(self.model, torch.nn.DataParallel) or \
               isinstance(self.model, torch.nn.parallel.DistributedDataParallel) \
            else self.model.state_dict()
        state = {
            "steps":
                self.stats.steps,
            "total_tokens":
                self.stats.total_tokens,
            "best_ckpt_score":
                self.stats.best_ckpt_score,
            "best_ckpt_iteration":
                self.stats.best_ckpt_iter,
            "model_state":
                model_state_dict,
            "optimizer_state":
                self.optimizer.state_dict(),
            "scheduler_state":
                self.scheduler.state_dict() if self.scheduler is not None else None,
            'amp_state':
                amp.state_dict() if self.fp16 else None,
            "epoch_no":
                self.epoch_no,
            'ddp':
                self.ddp,
            "scores_queue":
                self.stats.previous_scores_queue
        }

        torch.save(state, model_path)
        symlink_target = "{}.ckpt".format(self.stats.steps)
        if new_best:
            if len(self.ckpt_queue) == self.ckpt_queue.maxlen:
                to_delete = self.ckpt_queue.popleft()  # delete oldest ckpt
                try:
                    os.remove(to_delete)
                except FileNotFoundError:
                    logger.warning(
                        "Wanted to delete old checkpoint %s but "
                        "file does not exist.", to_delete)

            self.ckpt_queue.append(model_path)

            best_path = "{}/best.ckpt".format(self.model_dir)
            try:
                # create/modify symbolic link for best checkpoint
                symlink_update(symlink_target, best_path)
            except OSError:
                # overwrite best.ckpt
                torch.save(state, best_path)

        if self.save_latest_checkpoint:
            last_path = "{}/latest.ckpt".format(self.model_dir)
            previous_path = latest_checkpoint_update(symlink_target, last_path)
            # If the last ckpt is in the ckpt_queue, we don't want to delete it.
            can_delete = True
            for ckpt_path in self.ckpt_queue:
                if pathlib.Path(ckpt_path).resolve() == previous_path:
                    can_delete = False
                    break
            if can_delete and previous_path is not None:
                os.remove(previous_path)

    def init_from_checkpoint(self,
                             path: str,
                             reset_best_ckpt: bool = False,
                             reset_scheduler: bool = False,
                             reset_optimizer: bool = False,
                             reset_iter_state: bool = False) -> None:
        """
        Initialize the trainer from a given checkpoint file.

        This checkpoint file contains not only model parameters, but also
        scheduler and optimizer states, see `self._save_checkpoint`.

        :param path: path to checkpoint
        :param reset_best_ckpt: reset tracking of the best checkpoint,
                                use for domain adaptation with a new dev
                                set or when using a new metric for fine-tuning.
        :param reset_scheduler: reset the learning rate scheduler, and do not
                                use the one stored in the checkpoint.
        :param reset_optimizer: reset the optimizer, and do not use the one
                                stored in the checkpoint.
        :param reset_iter_state: reset the sampler's internal state and do not
                                use the one stored in the checkpoint.
        """
        logger.info("Loading model from %s", path)
        model_checkpoint = load_checkpoint(path=path, use_cuda=self.use_cuda, ddp=self.ddp, rank=self.rank)

        # restore model and optimizer parameters
        self.model.load_state_dict(model_checkpoint["model_state"])
        model.ddp = model_checkpoint.get('ddp', False)
        if not reset_optimizer:
            self.optimizer.load_state_dict(model_checkpoint["optimizer_state"])
        else:
            logger.info("Reset optimizer.")

        if not reset_scheduler:
            if model_checkpoint["scheduler_state"] is not None and \
                    self.scheduler is not None:
                self.scheduler.load_state_dict(
                    model_checkpoint["scheduler_state"])
        else:
            logger.info("Reset scheduler.")

        # restore counts
        self.stats.steps = model_checkpoint["steps"]
        self.stats.total_tokens = model_checkpoint["total_tokens"]
        self.stats.previous_scores_queue = model_checkpoint["scores_queue"]

        if not reset_best_ckpt:
            self.stats.best_ckpt_score = model_checkpoint["best_ckpt_score"]
            self.stats.best_ckpt_iter = model_checkpoint["best_ckpt_iteration"]
        else:
            logger.info("Reset tracking of the best checkpoint.")

        # if (not reset_iter_state
        #         and model_checkpoint.get('train_iter_state', None) is not None):
        #     self.train_iter_state = model_checkpoint["train_iter_state"]
        # reset the epoch
        self.epoch_no = model_checkpoint['epoch_no']
        # move parameters to cuda
        if self.use_cuda and not self.ddp:
            self.model.to(self.device)

        # fp16
        if self.fp16 and model_checkpoint.get("amp_state", None) is not None:
            amp.load_state_dict(model_checkpoint['amp_state'])

    def make_small_data(self, data,
                        small_data_size=30, small_epochs=2, batch_size_dividor=5):
        logger.info(f"The lenght of small data is {small_data_size}")
        self.batch_size = small_data_size // batch_size_dividor
        # self.epochs = small_epochs
        self.validation_freq = 10000

        subset_data, _ = torch.utils.data.random_split(data,
                                                       [small_data_size, len(data) - small_data_size],
                                                       torch.Generator().manual_seed(
                                                           self.train_config.get("random_seed", 42)))
        subset_data.src_vocab = data.src_vocab
        subset_data.trg_vocab = data.trg_vocab
        return subset_data

    def init_ddp(self, gpu, train_data, child_conn):
        logger.info(f"Got into training, gpu is {gpu}")
        self.rank = self.node_nr * self.n_gpu + gpu
        dist.init_process_group(
            backend='nccl',  # gloo
            init_method='env://',
            world_size=self.world_size,
            rank=self.rank
        )
        self.device = torch.device(gpu)
        self.model.to(self.device)
        logger.info(f"Model wrapped")
        # init stats for each process:

        if "load_model" in self.train_config.keys():
            self.init_from_checkpoint(
                self.train_config["load_model"],
                reset_best_ckpt=self.train_config.get("reset_best_ckpt", False),
                reset_scheduler=self.train_config.get("reset_scheduler", False),
                reset_optimizer=self.train_config.get("reset_optimizer", False),
                reset_iter_state=self.train_config.get("reset_iter_state", False))
        self.model = _DistributedDataParallel(self.model, device_ids=[gpu])

        if self.distributed_batch_sampler:
            self.batch_sampler = DistributedBatchSamplerSimilarLength(train_data, self.batch_size)
        else:
            self.train_sampler = DistributedSampler(train_data, num_replicas=self.world_size,
                                                    rank=self.rank)

        if self.rank == 0:
            self.tb_writer = SummaryWriter(log_dir=self.model_dir + "/tensorboard/")
            if self.use_wandb:
                self.init_wandb(self.config)
        self.child_conn = child_conn

    def _log_training(self, batch_loss, log_dict):
        # tensorflow
        self.tb_writer.add_scalar("train/train_batch_loss",
                                  batch_loss, self.stats.steps)
        # wandb
        if self.use_wandb:
            wandb.log({"train/train_batch_loss_normalised": batch_loss})
            wandb.log({"train/nll": log_dict['nll']})

        if log_dict["utility_term"] is not None:
            # special, since it's a histogram
            u_h = log_dict['u_h']
            del log_dict["u_h"]
            log_dict = {f"train/{key}": value for key, value in log_dict.items()}

            for stat_name, stat_value in log_dict.items():
                self.tb_writer.add_scalar(stat_name, stat_value, self.stats.steps)
            self.tb_writer.add_histogram("train/utilities_histogram", u_h,
                                         self.stats.steps)
            if self.use_wandb:
                logger.info(f'u_h is {u_h.shape}')
                log_dict["train/utilities_histogram"] = wandb.Histogram(u_h)
                # print(f"is u_h in log_dict? {log_dict['train/utilities_histogram']}")
                wandb.log(log_dict)

    # pylint: disable=unnecessary-comprehension
    # pylint: disable=too-many-branches
    # pylint: disable=too-many-statements
    def train_and_validate(self, gpu: int = 0, train_data: Dataset = None, valid_data: Dataset = None, child_conn=None) \
            -> None:
        """
        Train the model and validate it from time to time on the validation set.

        :param train_data: training data
        :param valid_data: validation data
        """

        logger.info(f'Initialised {"gpu" if torch.cuda.is_available() else "cpu"} {gpu}')
        # if small test run, use subset of the data that is one batch
        if self.small_test_run:
            train_data = self.make_small_data(train_data)

        # if Distributed Data Parallel
        if self.ddp:
            self.init_ddp(gpu, train_data, child_conn)
        else:
            self.batch_sampler = BatchSamplerSimilarLength(train_data, self.batch_size, shuffle=self.shuffle)
            self.tb_writer = SummaryWriter(log_dir=self.model_dir + "/tensorboard/")

        self.dataloader = make_dataloader(train_data,
                                          batch_size=self.batch_size,
                                          batch_type=self.batch_type,
                                          train=True,
                                          shuffle=self.shuffle if self.train_sampler is None else False,
                                          sampler=self.train_sampler,
                                          batch_sampler=self.batch_sampler)

        #################################################################
        # simplify accumulation logic:
        #################################################################
        # for epoch in range(epochs):
        #     self.model.zero_grad()
        #     epoch_loss = 0.0
        #     batch_loss = 0.0
        #     for i, batch in enumerate(iter(self.dataloader)):
        #
        #         # gradient accumulation:
        #         # loss.backward() inside _train_step()
        #         batch_loss += self._train_step(inputs)
        #
        #         if (i + 1) % self.batch_multiplier == 0:
        #             self.optimizer.step()     # update!
        #             self.model.zero_grad()    # reset gradients
        #             self.steps += 1           # increment counter
        #
        #             epoch_loss += batch_loss  # accumulate batch loss
        #             batch_loss = 0            # reset batch loss
        #
        #     # leftovers are just ignored.
        #################################################################

        logger.info(
            "Train stats:\n"
            "\tdevice: %s\n"
            "\tn_gpu: %d\n"
            "\t16-bits training: %r\n"
            "\tgradient accumulation: %d\n"
            "\tbatch size per device: %d\n"
            "\ttotal batch size (w. parallel & accumulation): %d", self.device,
            self.n_gpu, self.fp16, self.batch_multiplier, self.batch_size //
                                                          self.n_gpu if self.n_gpu > 1 and not self.ddp else self.batch_size,
            self.batch_size * self.batch_multiplier)

        # if we don't train, for debug purposes
        if self.epochs == 0:
            epoch_no = 0
            self._save_checkpoint(True)

        for epoch_no in range(self.epoch_no, self.epochs):
            logger.info("EPOCH %d", epoch_no + 1)

            if self.scheduler is not None and self.scheduler_step_at == "epoch":
                self.scheduler.step(epoch=epoch_no)

            self.model.train()

            # Reset statistics for each epoch.

            start = time.time()  # if not self.ddp or self.rank == 0 else 0
            epoch_duration = 0
            total_valid_duration = 0
            start_tokens = self.stats.total_tokens
            self.model.zero_grad()
            epoch_loss = 0
            batch_loss = 0
            self.epoch_no = epoch_no
            # update the epoch so distributed can have a different order
            if self.ddp:
                if self.distributed_batch_sampler:
                    self.batch_sampler.set_epoch(self.epoch_no)
                else:
                    self.train_sampler.set_epoch(self.epoch_no)

            for i, batch in enumerate(self.dataloader):
                # print(batch[1], batch[-1])
                # create a Batch object from torch batch
                batch = self.batch_class(batch, self.model.pad_index,
                                         use_cuda=self.use_cuda, device=self.device)
                # print(batch.src.shape, batch.src_length)

                # get batch loss
                batch_loss_iter, log_dict = self._train_step(batch)
                batch_loss += batch_loss_iter
                # update!
                if (i + 1) % self.batch_multiplier == 0:
                    # clip gradients (in-place)
                    if self.clip_grad_fun is not None:
                        if self.fp16:
                            self.clip_grad_fun(
                                params=amp.master_params(self.optimizer))
                        else:
                            self.clip_grad_fun(params=self.model.parameters())

                    # make gradient step
                    self.optimizer.step()

                    # decay lr
                    if self.scheduler is not None \
                            and self.scheduler_step_at == "step":
                        self.scheduler.step()

                    # reset gradients
                    self.model.zero_grad()

                    # increment step counter
                    self.stats.steps += 1

                    # log learning progress
                    # todo log more things here
                    if self.stats.steps % self.logging_freq == 0 or self.small_test_run:
                        if not self.ddp or self.rank == 0:
                            self._log_training(batch_loss, log_dict)
                        elapsed = time.time() - start - total_valid_duration

                        elapsed_tokens = self.stats.total_tokens - start_tokens
                        logger.info(
                            f"Epoch {epoch_no + 1}, Step: {self.stats.steps}, Batch Loss: {batch_loss:.6f}, "
                            f"Tokens per Sec: {elapsed_tokens / elapsed :.0f}, "
                            f"LR: {self.optimizer.param_groups[0]['lr']:.6f}, rank {0 if self.rank is None else self.rank}"
                        )

                        epoch_duration += elapsed

                        start = time.time()  # if not self.ddp or self.rank == 0 else 0
                        total_valid_duration = 0
                        start_tokens = self.stats.total_tokens

                    # Only add complete loss of full mini-batch to epoch_loss
                    epoch_loss += batch_loss  # accumulate epoch_loss
                    batch_loss = 0  # rest batch_loss

                    # validate on the entire dev set
                    if self.stats.steps % self.validation_freq == 0:
                        valid_duration = self._validate(valid_data, epoch_no)
                        if self.track_mbr:
                            mbr_valid_duration = self._validate(valid_data, epoch_no, True)
                            valid_duration += mbr_valid_duration
                        total_valid_duration += valid_duration

                if self.stats.stop:
                    break

            else:
                # if not self.ddp or self.rank == 0:
                epoch_duration = time.time() - start - total_valid_duration + epoch_duration

                logger.info(
                    f"End of epoch {epoch_no + 1}, it took {epoch_duration:.3f}. Epoch loss is {epoch_loss:.4f}. "
                    f"Rank is {0 if self.rank is None else self.rank}")

            # validate at the end of epoch if small
            if self.small_test_run:
                valid_duration = self._validate(valid_data, epoch_no)
                if self.track_mbr:
                    mbr_valid_duration = self._validate(valid_data, epoch_no, True)
                    valid_duration += mbr_valid_duration
                    total_valid_duration += valid_duration

            if self.stats.stop:
                if self.stats.early_stopping:
                    logger.info(f'Training ended due to early stopping.'
                                f' {self.early_stopping_metric} did not improve on the validation set'
                                f'for the last {self.early_stopping_patience} evaluations.')
                else:
                    logger.info('Training ended since minimum lr %f was reached.',
                                self.learning_rate_min)
                break

        else:
            logger.info('Training ended after %3d epochs.', epoch_no + 1)
        logger.info('Best validation result (greedy) at step %8d: %6.2f %s.',
                    self.stats.best_ckpt_iter, self.stats.best_ckpt_score,
                    self.early_stopping_metric)

        # send the best ckpt to the parent
        if self.child_conn is not None:
            logger.info(f"child connection {self.rank}")
            self.child_conn.send(self.stats.best_ckpt_iter)
            logger.info(f"child connection done {self.rank}")

        if not self.ddp or self.rank == 0:
            self.tb_writer.close()  # close Tensorboard writer
            if self.use_wandb:
                wandb.finish()

    def _train_step(self, batch: Batch) -> (Tensor, dict):
        """
        Train the model on one batch: Compute the loss.

        :param batch: training batch
        :return: loss for batch (sum), log dictionary
        """
        # reactivate training
        self.model.train()
        # get loss
        batch_loss, log_dict, _, _ = self.model(return_type="loss",
                                                **{"batch": batch,
                                                   "utility_regularising": self.utility_regularising,
                                                   'utility_type': self.utility_type,
                                                   **vars(batch)})

        # sum multi-gpu losses
        if self.n_gpu > 1 and not self.ddp:
            batch_loss = batch_loss.sum()

        # normalize batch loss
        if self.normalization == "batch":
            normalizer = batch.nseqs
        elif self.normalization == "tokens":
            normalizer = batch.ntokens
        elif self.normalization == "none":
            normalizer = 1
        else:
            raise NotImplementedError("Only normalize by 'batch' or 'tokens' "
                                      "or summation of loss 'none' implemented")

        norm_batch_loss = batch_loss / normalizer

        if self.n_gpu > 1 and not self.ddp:
            norm_batch_loss = norm_batch_loss / self.n_gpu

        if self.batch_multiplier > 1:
            norm_batch_loss = norm_batch_loss / self.batch_multiplier

        # accumulate gradients
        if self.fp16:
            with amp.scale_loss(norm_batch_loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            norm_batch_loss.backward()

        # increment token counter
        self.stats.total_tokens += batch.ntokens

        return norm_batch_loss.item(), log_dict

    def _synch_reduce_ddp(self, score, reduce_type="mean"):
        """
        Synchronises validation scores between different gpus while doing DDP.
        Args:
            score (float/int): score to be synchronised.
            reduce_type (str): how to reduce the score across the machine.
                                One out of ["mean", "sum", "prod"]

        Returns:
            float representing the reduced score

        """
        valid_reduce_types = ["mean", "sum", "prod"]
        assert reduce_type in valid_reduce_types, \
            f"Please make sure to use reduce_type argument from {valid_reduce_types} for synchronisation."
        # if the score is tensor, clone it, otherwise create a tensor of it
        if type(score) == torch.Tensor:
            tensor_score = score.detach().clone().to(self.device)
        else:
            tensor_score = torch.tensor(score, device=self.device, dtype=torch.float)
        # create a list that will hold synchronised score
        score_list = [torch.zeros_like(tensor_score) for _ in range(self.world_size)]
        # put all the tensors across the machines in the appropriate positions
        dist.all_gather(score_list, tensor_score)

        # stack all the machines
        stacked_list = torch.stack(score_list)

        # reduce to single score
        if reduce_type == "mean":
            reduced = torch.mean(stacked_list)
        elif reduce_type == "sum":
            reduced = torch.sum(stacked_list)
        # for perplexity = e^{-logP/N} = \prod_i^W e^{-logP_i/N_i} = e^{\sum_i^W -logP_i/N_i}
        elif reduce_type == "prod":
            reduced = torch.prod(stacked_list)
        return reduced.item()

    # def _synch_reduce_ddp(self, score, reduce_type="mean"):
    #     if type(score) == torch.Tensor:
    #         tensor_score = score.detach().clone().to(self.device)
    #     else:
    #         tensor_score = torch.tensor(score, device=self.device, dtype=torch.float)
    #     dist.all_reduce(tensor_score)
    #     tensor_score = tensor_score / self.world_size
    #     return tensor_score

    def _validate(self, valid_data, epoch_no, track_mbr=False):
        # search method for validation
        valid_type_str = 'mbr' if track_mbr else 'greedy'
        # based on what type of search to make decisions on
        # if mbr is being tracked, use that, otherwise, greedy will suffice
        if getattr(self, 'track_mbr', False):
            make_decision = True if track_mbr else False
        else:
            make_decision = False if track_mbr else True
        valid_start_time = time.time()
        valid_score, valid_loss, valid_ppl, valid_sources, \
        valid_sources_raw, valid_references, valid_hypotheses, \
        valid_hypotheses_raw, valid_attention_scores, valid_utility = \
            validate_on_data(
                batch_size=self.eval_batch_size,
                batch_class=self.batch_class,
                data=valid_data,
                eval_metric=self.eval_metric,
                level=self.level, model=self.model,
                use_cuda=self.use_cuda,
                max_output_length=self.max_output_length,
                compute_loss=True,
                beam_size=1,  # greedy validations
                batch_type=self.eval_batch_type,
                postprocess=True,  # always remove BPE for validation
                bpe_type=self.bpe_type,  # "subword-nmt" or "sentencepiece"
                sacrebleu=self.sacrebleu,  # sacrebleu options
                n_gpu=self.n_gpu,
                small_test_run=self.small_test_run,
                mbr=True if track_mbr else False,
                utility_type=self.utility_type,
                rank=self.rank,
                world_size=self.world_size
            )

        # synchronise+reduce valid scores between the processess
        if self.ddp:
            # todo normalise loss in predict
            # todo compute bleu + sentence level utility histogram validation
            # utility average
            valid_loss = self._synch_reduce_ddp(valid_loss, reduce_type="mean")
            if self.eval_metric != '':
                valid_score = self._synch_reduce_ddp(valid_score, reduce_type="mean")
            valid_ppl = self._synch_reduce_ddp(valid_ppl, reduce_type="mean")
            valid_utility = self._synch_reduce_ddp(valid_utility, reduce_type="mean")

        if not self.ddp or self.rank == 0:
            for name, score in zip(["valid_loss", "valid_score", "valid_ppl", "valid_utility"],
                                   [valid_loss, valid_score, valid_ppl, valid_utility]):
                # don't log, if not computed
                if self.eval_metric == '' and name == 'valid_score':
                    continue
                self.tb_writer.add_scalar(f"valid/{valid_type_str}/{name}", score,
                                          self.stats.steps)
                if self.use_wandb:
                    wandb.log({f"valid/{valid_type_str}/{name}": score})

        if self.early_stopping_metric == "loss":
            ckpt_score = valid_loss
        elif self.early_stopping_metric in ["ppl", "perplexity"]:
            ckpt_score = valid_ppl
        elif self.early_stopping_metric == "utility":
            ckpt_score = valid_utility
        else:
            ckpt_score = valid_score

        if self.scheduler is not None and self.scheduler_step_at == "validation" and make_decision:
            self.scheduler.step(ckpt_score)

        # early stopping of the training
        if self.early_stopping and make_decision:
            self.stats.early_stopping_step(ckpt_score)

        # checkpointing
        new_best = False
        if make_decision and self.stats.is_best(ckpt_score) and (not self.ddp or self.rank == 0):
            self.stats.best_ckpt_score = ckpt_score
            self.stats.best_ckpt_iter = self.stats.steps
            logger.info('Hooray! New best validation result [%s]!',
                        self.early_stopping_metric)
            if self.ckpt_queue.maxlen > 0:
                logger.info("Saving new checkpoint.")
                new_best = True
                self._save_checkpoint(new_best)
        elif self.save_latest_checkpoint and (not self.ddp or self.rank == 0) and make_decision:
            self._save_checkpoint(new_best)

        if not self.ddp or self.rank == 0:
            # append to validation report
            self._add_report(valid_score=valid_score,
                             valid_loss=valid_loss,
                             valid_ppl=valid_ppl,
                             eval_metric=self.eval_metric,
                             new_best=new_best,
                             utility=valid_utility,
                             mbr=track_mbr)

        if not self.small_test_run:
            self._log_examples(sources_raw=[v for v in valid_sources_raw],
                               sources=valid_sources,
                               hypotheses_raw=valid_hypotheses_raw,
                               hypotheses=valid_hypotheses,
                               references=valid_references)

        valid_duration = time.time() - valid_start_time  # if not self.ddp or self.rank == 0 else 0
        if not self.ddp or self.rank == 0:
            logger.info(
                f'Validation result (%s) at epoch %3d, '
                'step %8d: %s: %6.2f, loss: %8.4f, ppl: %8.4f,  utility %s:%8.2f '
                'duration: %.4fs, rank %2d', valid_type_str, epoch_no + 1, self.stats.steps, self.eval_metric,
                valid_score, valid_loss, valid_ppl, self.utility_type, valid_utility, valid_duration,
                0 if self.rank is None else self.rank)

        # store validation set outputs
        self._store_outputs(valid_hypotheses, mbr=track_mbr)

        # store attention plots for selected valid sentences
        if valid_attention_scores and make_decision:
            store_attention_plots(attentions=valid_attention_scores,
                                  targets=valid_hypotheses_raw,
                                  sources=[s for s in valid_data.src],
                                  indices=self.log_valid_sents,
                                  output_prefix="{}/att.{}".format(
                                      self.model_dir, self.stats.steps),
                                  tb_writer=self.tb_writer,
                                  steps=self.stats.steps)

        return valid_duration

    def _add_report(self,
                    valid_score: float,
                    valid_ppl: float,
                    valid_loss: float,
                    eval_metric: str,
                    utility: float = 0,
                    new_best: bool = False,
                    mbr: bool = False) -> None:
        """
        Append a one-line report to validation logging file.

        :param valid_score: validation evaluation score [eval_metric]
        :param valid_ppl: validation perplexity
        :param valid_loss: validation loss (sum over whole validation set)
        :param eval_metric: evaluation metric, e.g. "bleu"
        :param new_best: whether this is a new best model
        """
        current_lr = -1
        # ignores other param groups for now
        for param_group in self.optimizer.param_groups:
            current_lr = param_group['lr']

        if current_lr < self.learning_rate_min:
            self.stats.stop = True

        with open(self.valid_report_file, 'a') as opened_file:
            opened_file.write(
                "Steps: {}\tLoss: {:.5f}\tPPL: {:.5f}\t{}: {:.5f}\t"
                "Utility {}:{:.5f}\tLR: {:.8f}\t{}\t{}\n".format(self.stats.steps, valid_loss,
                                                                 valid_ppl, eval_metric, valid_score,
                                                                 self.utility_type, utility,
                                                                 current_lr, "*" if new_best else "",
                                                                 "mbr" if mbr else "greedy"))

    def _log_parameters_list(self) -> None:
        """
        Write all model parameters (name, shape) to the log.
        """

        def req_grad(p): return p.requires_grad

        model_parameters = filter(req_grad,
                                  self.model.parameters())
        n_params = sum([np.prod(p.size()) for p in model_parameters])
        logger.info("Total params: %d", n_params)
        trainable_params = [
            n for (n, p) in self.model.named_parameters() if p.requires_grad
        ]
        logger.debug("Trainable parameters: %s", sorted(trainable_params))

    def _log_examples(self,
                      sources: List[str],
                      hypotheses: List[str],
                      references: List[str],
                      sources_raw: List[List[str]] = None,
                      hypotheses_raw: List[List[str]] = None,
                      references_raw: List[List[str]] = None) -> None:
        """
        Log a the first `self.log_valid_sents` sentences from given examples.

        :param sources: decoded sources (list of strings)
        :param hypotheses: decoded hypotheses (list of strings)
        :param references: decoded references (list of strings)
        :param sources_raw: raw sources (list of list of tokens)
        :param hypotheses_raw: raw hypotheses (list of list of tokens)
        :param references_raw: raw references (list of list of tokens)
        """
        for p in self.log_valid_sents:

            if p >= len(sources):
                continue

            logger.info("Example #%d", p)

            if sources_raw is not None:
                logger.debug("\tRaw source:     %s", sources_raw[p])
            if references_raw is not None:
                logger.debug("\tRaw reference:  %s", references_raw[p])
            if hypotheses_raw is not None:
                logger.debug("\tRaw hypothesis: %s", hypotheses_raw[p])

            logger.info("\tSource:     %s", sources[p])
            logger.info("\tReference:  %s", references[p])
            logger.info("\tHypothesis: %s", hypotheses[p])

    def _store_outputs(self, hypotheses: List[str], mbr=False) -> None:
        """
        Write current validation outputs to file in `self.model_dir.`

        :param hypotheses: list of strings
        """

        current_valid_output_file = "{}/{}_{}.hyps".format(self.model_dir,
                                                           self.stats.steps,
                                                           mbr)

        with open(current_valid_output_file, 'w') as opened_file:
            for hyp in hypotheses:
                opened_file.write("{}\n".format(hyp))

    class TrainStatistics:
        def __init__(self,
                     steps: int = 0,
                     stop: bool = False,
                     total_tokens: int = 0,
                     best_ckpt_iter: int = 0,
                     best_ckpt_score: float = np.inf,
                     minimize_metric: bool = True,
                     early_stopping_patience=5) -> None:
            # global update step counter
            self.steps = steps
            # stop training if this flag is True
            # by reaching learning rate minimum or due to early stopping
            self.stop = stop
            # number of total tokens seen so far
            self.total_tokens = total_tokens
            # store iteration point of best ckpt
            self.best_ckpt_iter = best_ckpt_iter
            # initial values for best scores
            self.best_ckpt_score = best_ckpt_score
            # minimize or maximize score
            self.minimize_metric = minimize_metric
            if self.minimize_metric:
                self.comparison = operator.lt
            else:
                self.comparison = operator.gt

            # early stopping stats
            self.early_stopping_patience = early_stopping_patience
            if self.minimize_metric:
                self.previous_scores_queue = [np.inf] * self.early_stopping_patience
            else:
                self.previous_scores_queue = [-np.inf] * self.early_stopping_patience
            # reason for stopping
            self.early_stopping = False

        def is_best(self, score):
            is_best = self.comparison(score, self.best_ckpt_score)
            return is_best

        def early_stopping_step(self, score):
            # remove the oldest element in the queue
            self.previous_scores_queue.pop(0)
            # append the score
            self.previous_scores_queue.append(score)
            # shall we stop?
            self.early_stopping = True
            # compare each element sequentially. if it does not improve, early stop
            for i, previous_score in enumerate(self.previous_scores_queue[:-1]):
                next_score = self.previous_scores_queue[i + 1]
                is_improving = self.comparison(next_score, previous_score)
                if is_improving:
                    self.early_stopping = False
            # stop
            if self.early_stopping:
                self.stop = True


def train(cfg_file: str) -> None:
    """
    Main training function. After training, also test on test data if given.

    :param cfg_file: path to configuration yaml file
    """

    cfg = load_config(cfg_file)

    # make logger
    model_dir = make_model_dir(cfg["training"]["model_dir"],
                               overwrite=cfg["training"].get(
                                   "overwrite", False))
    _ = make_logger(model_dir, mode="train")  # version string returned
    # TODO: save version number in model checkpoints

    # set the random seed
    set_seed(seed=cfg["training"].get("random_seed", 42))

    # load the data

    train_data, dev_data, test_data, src_vocab, trg_vocab = load_data(
        data_cfg=cfg["data"])

    # build an encoder-decoder model
    model = build_model(cfg["model"], src_vocab=src_vocab, trg_vocab=trg_vocab)

    # for training management, e.g. early stopping and model selection
    trainer = TrainManager(model=model, config=cfg)
    # store copy of original training config in model dir
    shutil.copy2(cfg_file, model_dir + "/config.yaml")

    # log all entries of config
    log_cfg(cfg)
    if not cfg["training"].get("skip_vocab", True):
        log_data_info(train_data=train_data,
                      valid_data=dev_data,
                      test_data=test_data,
                      src_vocab=src_vocab,
                      trg_vocab=trg_vocab)

    logger.info(str(model))

    # store the vocabs
    src_vocab_file = "{}/src_vocab.txt".format(cfg["training"]["model_dir"])
    src_vocab.to_file(src_vocab_file)
    trg_vocab_file = "{}/trg_vocab.txt".format(cfg["training"]["model_dir"])
    trg_vocab.to_file(trg_vocab_file)
    # train the model
    if cfg["training"].get("ddp", False) and trainer.n_gpu > 0:
        parent_conn, child_conn = mp.Pipe()
        logger.info("Training using multiple GPUs")
        spawn_multiprocess(trainer.train_and_validate, (train_data, dev_data, child_conn))
        last_checkpoint = parent_conn.recv()
        i = 0
        while parent_conn.poll():
            logger.info(f"Parent connection {i} got {parent_conn.recv()}, last checkpoint is {last_checkpoint}")
            i += 1
        trainer.stats.best_ckpt_iter = last_checkpoint
    else:
        trainer.train_and_validate(train_data=train_data, valid_data=dev_data)

    # predict with the best model on validation and test
    # (if test data is available)
    ckpt = "{}/{}.ckpt".format(model_dir, trainer.stats.best_ckpt_iter)
    output_name = "{:08d}.hyps".format(trainer.stats.best_ckpt_iter)
    output_path = os.path.join(model_dir, output_name)
    # if after checkpointing things did not improve, then we can use the checkpoint
    if not os.path.exists(output_path) and 'load_model' in trainer.train_config:
        logger.info("After checkpointing, the model did not improve. Using checkpoint save")
        output_path = trainer.train_config['load_model']
    datasets_to_test = {
        "dev": dev_data,
        "test": test_data,
        "src_vocab": src_vocab,
        "trg_vocab": trg_vocab
    }
    logger.info(f"checkpoint name {ckpt}")
    test(cfg_file,
         ckpt=ckpt,
         output_path=output_path,
         datasets=datasets_to_test)


def spawn_multiprocess(train_fn, args):
    num_gpus = torch.cuda.device_count()
    # world_size = num_gpus * num_nodes
    os.environ['MASTER_ADDR'] = '127.0.0.1'  # lisa IP
    os.environ['MASTER_PORT'] = '9028'
    # os.environ['MKL_THREADING_LAYER']='GNU' # if numpy<=20.1
    mp.spawn(train_fn, nprocs=num_gpus, args=args)
    logger.info("Multiple GPUs are done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Joey-NMT')
    parser.add_argument("config",
                        default="configs/default.yaml",
                        type=str,
                        help="Training configuration file (yaml).")
    args = parser.parse_args()
    train(cfg_file=args.config)
