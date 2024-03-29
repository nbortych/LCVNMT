# coding: utf-8
"""
This modules holds methods for generating predictions from a model.
"""
import os
import sys
from typing import List, Optional
import logging
import itertools
import time
import editdistance

import numpy as np
import torch
import multiprocessing as mp
from multiprocessing.pool import ThreadPool
from torchtext.legacy.data import Field
from torch.utils.data import Dataset, Subset
# from torch.utils.data.distributed import DistributedSampler
import sacrebleu
from sacrebleu.metrics import BLEU, CHRF

import wandb
from joeynmt.helpers import bpe_postprocess, load_config, make_logger, \
    get_latest_checkpoint, load_checkpoint, store_attention_plots, debug_memory, repeat_batch
from joeynmt.metrics import bleu, chrf, token_accuracy, sequence_accuracy
from joeynmt.model import build_model, Model, _DataParallel
from joeynmt.search import run_batch
from joeynmt.batch import Batch
from joeynmt.data import load_data, make_dataloader, MonoDataset, DistributedEvalSampler
from joeynmt.constants import UNK_TOKEN, PAD_TOKEN, EOS_TOKEN
from joeynmt.vocabulary import Vocabulary

try:
    os.environ["BEER_HOME"]
except:
    os.environ["BEER_HOME"] = "./beer_2.0"

logger = logging.getLogger(__name__)

logger.info(f'Sacrebleu version {sacrebleu.__version__}')


# pylint: disable=too-many-arguments,too-many-locals,no-member
def validate_on_data(model: Model, data: Dataset,
                     batch_size: int,
                     use_cuda: bool, max_output_length: int,
                     level: str, eval_metric: Optional[str],
                     n_gpu: int,
                     batch_class: Batch = Batch,
                     compute_loss: bool = False,
                     beam_size: int = 1, beam_alpha: int = -1,
                     batch_type: str = "sentence",
                     postprocess: bool = True,
                     bpe_type: str = "subword-nmt",
                     sacrebleu_dict: dict = None,
                     sample: bool = False,
                     mbr: bool = False,
                     mbr_type: str = 'editdistance',
                     small_test_run: bool = False,
                     num_samples: int = 10,
                     save_utility_per_sentence: bool = False,
                     utility_type=None,
                     rank=0,
                     world_size=1,
                     utility_regularising_loss=False,
                     precompute_batch=False,
                     dynamic_max_output=False,
                     multi_utility=False) \
        -> (float, float, float, List[str], List[List[str]], List[str],
            List[str], List[List[str]], List[np.array]):
    """
    Generate translations for the given data.
    If `compute_loss` is True and references are given,
    also compute the loss.

    :param model: model module
    :param data: dataset for validation
    :param batch_size: validation batch size
    :param batch_class: class type of batch
    :param use_cuda: if True, use CUDA
    :param max_output_length: maximum length for generated hypotheses
    :param level: segmentation level, one of "char", "bpe", "word"
    :param eval_metric: evaluation metric, e.g. "bleu"
    :param n_gpu: number of GPUs
    :param compute_loss: whether to computes a scalar loss
        for given inputs and targets
    :param beam_size: beam size for validation.
        If <2 then greedy decoding (default).
    :param beam_alpha: beam search alpha for length penalty,
        disabled if set to -1 (default).
    :param batch_type: validation batch type (sentence or token)
    :param postprocess: if True, remove BPE segmentation from translations
    :param bpe_type: bpe type, one of {"subword-nmt", "sentencepiece"}
    :param sacrebleu_dict: sacrebleu_dict options
    :param sample: If True, non-greedy sampling during greedy decoding
    :param mbr:  If True, will use mbr for decoding

    :return:
        - current_valid_score: current validation score [eval_metric],
        - valid_loss: validation loss,
        - valid_ppl:, validation perplexity,
        - valid_sources: validation sources,
        - valid_sources_raw: raw validation sources (before post-processing),
        - valid_references: validation references,
        - valid_hypotheses: validation_hypotheses,
        - decoded_valid: raw validation hypotheses (before post-processing),
        - valid_attention_scores: attention scores for validation hypotheses
    """
    model.ddp = getattr(model, "ddp", False)
    if not model.ddp and isinstance(model, torch.nn.DataParallel):
        assert batch_size >= n_gpu, "batch_size must be bigger than n_gpu."
    if sacrebleu_dict is None:  # assign default value
        sacrebleu_dict = {"remove_whitespace": True, "tokenize": "13a"}
    if batch_size > 1000 and batch_type == "sentence":
        logger.warning(
            "WARNING: Are you sure you meant to work on huge batches like "
            "this? 'batch_size' is > 1000 for sentence-batching. "
            "Consider decreasing it or switching to"
            " 'eval_batch_type: token'.")

    # if small test run, use subset of the data that is one batch
    if small_test_run:
        SMALL_DATA_SIZE = 100

        val_subset_data, _ = torch.utils.data.random_split(data,
                                                           [SMALL_DATA_SIZE, len(data) - SMALL_DATA_SIZE],
                                                           torch.Generator().manual_seed(42))
        val_subset_data.src_vocab = data.src_vocab
        val_subset_data.trg_vocab = data.trg_vocab
        data = val_subset_data
        logger.info(f"The length of validation data for small test run is {len(data)}")
        # batch_size = len(data)

    if model.ddp:
        ddp_sampler = DistributedEvalSampler(data, num_replicas=world_size, rank=rank)
        ddp_indices = ddp_sampler.indices
    else:
        ddp_sampler = None

    valid_dataloader = make_dataloader(dataset=data, batch_size=batch_size,
                                       shuffle=False, train=False, sampler=ddp_sampler)
    valid_sources_raw = [src for src, _, trg, _ in data]
    pad_index = model.src_vocab.stoi[PAD_TOKEN]
    # disable dropout
    if not getattr(model, "bnn", False):
        model.eval()
    # initialise utility function just once
    if mbr:
        if utility_type == "beer":
            # logger.info(f"Number of cpu is {mp.cpu_count()} rank {rank}, world size {world_size}")
            utility_fn = [get_utility_fn(utility_type) for _ in range(mp.cpu_count() // world_size)]
        else:
            utility_fn = get_utility_fn(utility_type)
    else:
        utility_fn = None
        # don't track gradients during validation
    with torch.no_grad():
        all_outputs = []
        valid_attention_scores = []
        total_loss = 0
        total_ntokens = 0
        total_nseqs = 0
        encoder_hidden, encoder_output = None, None
        expected_utility_total = 0
        # gather subset of data
        for i, valid_batch in enumerate(valid_dataloader):
            # run as during training to get validation loss (e.g. xent)

            batch = batch_class(valid_batch, pad_index, use_cuda=use_cuda, device=model.device)
            # sort batch now by src length and keep track of order
            sort_reverse_index = batch.sort_by_src_length()

            # run as during training with teacher forcing
            if compute_loss and batch.trg is not None:
                batch_loss, _, encoder_output, encoder_hidden = model(return_type="loss", **{"batch": batch,
                                                                                             "utility_regularising": utility_regularising_loss,
                                                                                             'utility_type': utility_type,
                                                                                             **vars(batch),
                                                                                             "return_encoded": mbr and precompute_batch})

                if n_gpu > 1 and not model.ddp:
                    batch_loss = batch_loss.mean()  # average on multi-gpu
                total_loss += batch_loss
                total_ntokens += batch.ntokens
                total_nseqs += batch.nseqs

            if dynamic_max_output:
                max_output_length = model.loss_function.dynamic_match_sample_size(batch)
            # run as during inference to produce translations
            if not mbr:
                output, attention_scores = run_batch(
                    model=model, batch=batch, beam_size=beam_size,
                    beam_alpha=beam_alpha, max_output_length=max_output_length, sample=sample)
            else:
                # logger.info(f"Validation batch {i}/{len(valid_dataloader)}, rank {rank}")
                mbr_dict = mbr_decoding(model=model, batch=batch, max_output_length=max_output_length,
                                        compute_log_probs=False, need_grad=False,
                                        return_types=["h", "mean_batch_expected_uility_h"],
                                        num_samples=num_samples, mbr_type=mbr_type,
                                        utility_type=utility_type,
                                        utility_fn=utility_fn,
                                        encoded_batch=[encoder_output, encoder_hidden],
                                        small_test_run=small_test_run, rank=rank, world_size=world_size)

                output, expected_utility = mbr_dict['h'], mbr_dict['mean_batch_expected_uility_h']
                attention_scores = None
                expected_utility_total += expected_utility
            # sort outputs back to original order
            all_outputs.extend(output[sort_reverse_index])

            valid_attention_scores.extend(
                attention_scores[sort_reverse_index]
                if attention_scores is not None and type(attention_scores) != int else [])

        # get only the data on the current gpu
        if model.ddp:
            data = Subset(data, indices=ddp_indices)
        assert len(all_outputs) == len(data)
        expected_utility_mean = expected_utility_total / len(valid_dataloader)
        if compute_loss and total_ntokens > 0:
            # exponent of token-level negative log prob
            valid_ppl = torch.exp(total_loss / total_ntokens).item()
            # total validation loss
            valid_loss = total_loss.item() / len(valid_dataloader)

        else:
            valid_loss = -1
            valid_ppl = -1

        # decode back to symbols
        decoded_valid = model.trg_vocab.arrays_to_sentences(arrays=all_outputs,
                                                            cut_at_eos=True)

        # evaluate with metric on full dataset
        join_char = " " if level in ["word", "bpe"] else ""
        valid_sources_and_references = [(join_char.join(s), join_char.join(t)) for s, _, t, _ in data]
        valid_sources, valid_references = zip(*valid_sources_and_references)
        valid_hypotheses = [join_char.join(t) for t in decoded_valid]

        # post-process
        if level == "bpe" and postprocess:
            valid_sources = [bpe_postprocess(s, bpe_type=bpe_type)
                             for s in valid_sources]
            valid_references = [bpe_postprocess(v, bpe_type=bpe_type)
                                for v in valid_references]
            valid_hypotheses = [bpe_postprocess(v, bpe_type=bpe_type)
                                for v in valid_hypotheses]

        # if references are given, evaluate against them
        if valid_references:
            assert len(valid_hypotheses) == len(valid_references)

            current_valid_score = 0
            if eval_metric.lower() == 'bleu':
                # this version does not use any tokenization
                current_valid_score = bleu(
                    valid_hypotheses, valid_references,
                    tokenize=sacrebleu_dict["tokenize"])
            elif eval_metric.lower() == 'chrf':
                current_valid_score = chrf(valid_hypotheses, valid_references,
                                           remove_whitespace=sacrebleu_dict["remove_whitespace"])
            elif eval_metric.lower() == 'token_accuracy':
                current_valid_score = token_accuracy(  # supply List[List[str]]
                    # todo remove data.trg
                    list(decoded_valid), list(data.trg))
            elif eval_metric.lower() == 'sequence_accuracy':
                current_valid_score = sequence_accuracy(
                    valid_hypotheses, valid_references)
            else:
                current_valid_score = -1
            # compute utility
            if utility_type is not None and not multi_utility:
                reduced_utility, utility_per_sentence = get_utility_of_samples(utility_type, valid_hypotheses,
                                                                               valid_references, reduce_type="mean",
                                                                               save_utility_per_sentence=save_utility_per_sentence)
            elif multi_utility:
                reduced_utility, utility_per_sentence = [], []
                for utility in ["beer", "chrf", "bleu"]:
                    reduced, per_sentence = get_utility_of_samples(utility, valid_hypotheses,
                                                                   valid_references, reduce_type="mean",
                                                                   save_utility_per_sentence=save_utility_per_sentence)
                    reduced_utility.append(reduced)
                    utility_per_sentence.append(per_sentence)

            else:
                reduced_utility = 0
                utility_per_sentence = None
        else:
            current_valid_score = -1
            reduced_utility = 0
            utility_per_sentence = None
    # terminate open processes
    if type(utility_fn) == list and utility_type == "beer":
        for utility in utility_fn:
            utility.proc.terminate()
    elif utility_type == "beer" and utility_fn is not None:
        utility_fn.proc.terminate()

    return current_valid_score, valid_loss, valid_ppl, valid_sources, \
           valid_sources_raw, valid_references, valid_hypotheses, \
           decoded_valid, valid_attention_scores, reduced_utility, utility_per_sentence, expected_utility_mean


def parse_test_args(cfg, mode="test"):
    """
    parse test args
    :param cfg: config object
    :param mode: 'test' or 'translate'
    :return:
    """
    if "test" not in cfg["data"].keys():
        raise ValueError("Test data must be specified in config.")

    batch_size = cfg["training"].get(
        "eval_batch_size", cfg["training"].get("batch_size", 1))
    batch_type = cfg["training"].get(
        "eval_batch_type", cfg["training"].get("batch_type", "sentence"))
    use_cuda = (cfg["training"].get("use_cuda", False)
                and torch.cuda.is_available())
    device = torch.device("cuda" if use_cuda else "cpu")
    if mode == 'test':
        n_gpu = torch.cuda.device_count() if use_cuda else 0
        k = cfg["testing"].get("beam_size", 1)
        batch_per_device = batch_size * k // n_gpu if n_gpu > 1 else batch_size * k
        logger.info("Process device: %s, n_gpu: %d, "
                    "batch_size per device: %d (with beam_size)",
                    device, n_gpu, batch_per_device)
        eval_metric = cfg["training"]["eval_metric"]

    elif mode == 'translate':
        # in multi-gpu, batch_size must be bigger than n_gpu!
        n_gpu = 1 if use_cuda else 0
        logger.debug("Process device: %s, n_gpu: %d", device, n_gpu)
        eval_metric = ""

    level = cfg["data"]["level"]
    max_output_length = cfg["training"].get("max_output_length", None)

    # whether to use beam search for decoding, 0: greedy decoding
    if "testing" in cfg.keys():
        beam_size = cfg["testing"].get("beam_size", 1)
        beam_alpha = cfg["testing"].get("alpha", -1)
        postprocess = cfg["testing"].get("postprocess", True)
        bpe_type = cfg["testing"].get("bpe_type", "subword-nmt")
        sacrebleu = {"remove_whitespace": True, "tokenize": "13a"}
        if "sacrebleu" in cfg["testing"].keys():
            sacrebleu["remove_whitespace"] = cfg["testing"]["sacrebleu"] \
                .get("remove_whitespace", True)
            sacrebleu["tokenize"] = cfg["testing"]["sacrebleu"] \
                .get("tokenize", "13a")
        # whether we sample or greedy
        sample = cfg["testing"].get("sample", False)
        mbr = cfg["testing"].get("mbr", False)
        small_test_run = cfg["testing"].get("small_test_run", False)
        num_samples = cfg["testing"].get("num_samples", 10)
        mbr_type = cfg["testing"].get("mbr_type", 'editdistance')
        utility_type = cfg["testing"].get("utility", 'editdistance')
        save_utility_per_sentence = cfg["testing"].get("save_utility_per_sentence", False)
        all_decoding_types = cfg["testing"].get("all_decoding_types", False)
        multi_utility = cfg["testing"].get("multi_utility", False)
        multi_mbr = cfg["testing"].get("multi_mbr", False)
        only_test = cfg["testing"].get("only_test", False)


    else:
        beam_size = 1
        beam_alpha = -1
        postprocess = True
        bpe_type = "subword-nmt"
        sacrebleu = {"remove_whitespace": True, "tokenize": "13a"}
        sample = False
        mbr = False
        small_test_run = False
        num_samples = 1
        mbr_type = 'editdistance'
        utility_type = None
        save_utility_per_sentence = False
        all_decoding_types = False
        multi_utility = False
        multi_mbr = False
        only_test = False

    mbr_text, empty = f"{mbr_type}_MBR_", ""
    decoding_description = f"{'Greedy decoding' if (not sample and not mbr) else f'{mbr_text if mbr else empty}Sample decoding'}" if beam_size < 2 else \
        "Beam search decoding with beam size = {} and alpha = {}". \
            format(beam_size, beam_alpha)

    tokenizer_info = f"[{sacrebleu['tokenize']}]" \
        if eval_metric == "bleu" else ""

    return batch_size, batch_type, use_cuda, device, n_gpu, level, \
           eval_metric, max_output_length, beam_size, beam_alpha, \
           postprocess, bpe_type, sacrebleu, decoding_description, \
           tokenizer_info, sample, mbr, mbr_type, small_test_run, \
           num_samples, utility_type, save_utility_per_sentence, \
           all_decoding_types, multi_utility, multi_mbr, only_test


# pylint: disable-msg=logging-too-many-args
def test(cfg_file,
         ckpt: str,
         batch_class: Batch = Batch,
         output_path: str = None,
         save_attention: bool = False,
         datasets: dict = None,
         save_utility_per_sentence: bool = False) -> None:
    """
    Main test function. Handles loading a model from checkpoint, generating
    translations and storing them and attention plots.

    :param cfg_file: path to configuration file
    :param ckpt: path to checkpoint to load
    :param batch_class: class type of batch
    :param output_path: path to output
    :param datasets: datasets to predict
    :param save_attention: whether to save the computed attention weights
    """
    test_start_time = time.time()
    cfg = load_config(cfg_file)
    model_dir = cfg["training"]["model_dir"]
    from joeynmt.training import init_wandb
    init_wandb(cfg)

    if len(logger.handlers) == 0:
        _ = make_logger(model_dir, mode="test")  # version string returned

    # when checkpoint is not specified, take latest (best) from model dir
    if ckpt is None:
        ckpt = f"{model_dir}/best.ckpt"
        if not os.path.exists(ckpt):
            ckpt = get_latest_checkpoint(model_dir)
        try:
            step = ckpt.split(model_dir + "/")[1].split(".ckpt")[0]
        except IndexError:
            step = "best"
    logger.info(f"Loading checkpoint {step}")
    # parse test args
    batch_size, batch_type, use_cuda, device, n_gpu, level, eval_metric, \
    max_output_length, beam_size, beam_alpha, postprocess, \
    bpe_type, sacrebleu, decoding_description, tokenizer_info, \
    sample, mbr, mbr_type, small_test_run, num_samples, utility_type, \
    save_utility_per_sentence, all_decoding_types, multi_utility, multi_mbr, \
    only_test = parse_test_args(cfg, mode="test")

    # load the data
    if datasets is None:
        _, dev_data, test_data, src_vocab, trg_vocab = load_data(
            data_cfg=cfg["data"], datasets=["dev", "test"])
        data_to_predict = {"test": test_data}
        if not only_test:
            data_to_predict["dev"] = dev_data
    else:  # avoid to load data again
        data_to_predict = {"test": datasets["test"]}
        if not only_test:
            data_to_predict["dev"] = datasets["dev"]
        src_vocab = datasets["src_vocab"]
        trg_vocab = datasets["trg_vocab"]

    # load model state from disk
    model_checkpoint = load_checkpoint(ckpt, use_cuda=use_cuda)

    # build model and load parameters into it
    model = build_model(cfg["model"], src_vocab=src_vocab, trg_vocab=trg_vocab)
    model.load_state_dict(model_checkpoint["model_state"])
    model.device = device

    if use_cuda:
        model.to(device)

    if all_decoding_types:
        decoding_types = ["beam", "greedy", "mbr"]
    elif multi_mbr:
        decoding_types = ["mbr_10", "mbr_20","mbr_80"]
    else:
        decoding_types = [None]


    # multi-gpu eval
    # todo fix multigpu (DDP has problems with returning dict + if data is not devisable by n_gpu)
    if n_gpu > 1 and not isinstance(model, torch.nn.DataParallel) and not model_checkpoint['ddp']:
        logger.info("Got into multigpu during testing")
        model = _DataParallel(model)

    for data_set_name, data_set in data_to_predict.items():
        if data_set is None:
            continue

        dataset_file = cfg["data"][data_set_name] + "." + cfg["data"]["trg"]
        logger.info("Decoding on %s set (%s)...", data_set_name, dataset_file)
        for decoding_type in decoding_types:
            if decoding_type == None:
                pass
            else:
                logger.info(f"Decoding using {decoding_type} decoding.")
                if decoding_type == "beam":
                    beam_size = 5
                    mbr = False
                elif decoding_type == "greedy":
                    beam_size = 1
                    mbr = False
                elif decoding_type == "mbr":
                    beam_size = 1
                    mbr = True
                elif decoding_type.split("_")[0] == "mbr":
                    beam_size = 1
                    mbr = True
                    num_samples = int(decoding_type.split("_")[1])

            # pylint: disable=unused-variable
            score, loss, ppl, sources, sources_raw, references, hypotheses, \
            hypotheses_raw, attention_scores, utility, utility_per_sentence, expected_utility_mean = validate_on_data(
                model, data=data_set, batch_size=batch_size,
                batch_class=batch_class, batch_type=batch_type, level=level,
                max_output_length=max_output_length, eval_metric=eval_metric,
                use_cuda=use_cuda, compute_loss=False, beam_size=beam_size,  # 5 beam_size
                beam_alpha=beam_alpha, postprocess=postprocess,
                bpe_type=bpe_type, sacrebleu_dict=sacrebleu, n_gpu=n_gpu, sample=sample, mbr=mbr,  # mbr
                small_test_run=small_test_run, num_samples=num_samples, mbr_type=mbr_type,
                save_utility_per_sentence=save_utility_per_sentence, utility_type=utility_type,
                multi_utility=multi_utility)
            # pylint: enable=unused-variable

            # if "trg" in data_set.fields:
            logger.info("%4s %s%s: %6.2f [%s]",
                        data_set_name, eval_metric, tokenizer_info,
                        score, decoding_description)

            log_names, log_values = ["score", 'loss', "ppl", "expected_utility_mean"], [score, loss, ppl,
                                                                                        expected_utility_mean]

            if not multi_utility:
                log_names.extend([f"utility_{utility_type}", "utility_per_sentence"])
                log_values.extend([utility, wandb.Histogram(utility_per_sentence)])
            else:
                log_names.extend(["utility_beer", "utility_chrf", "utility_bleu", "utility_per_sentence_beer",
                                  "utility_per_sentence_chrf", "utility_per_sentence_bleu"])
                log_values.extend(utility)
                histograms = list(map(wandb.Histogram, utility_per_sentence))
                log_values.extend(histograms)

            log_dict = {f"test/{decoding_type if decoding_type is not None else 'mbr'}/{step if step == 'best' else 'converged'}/{key}": value for key, value
                        in
                        zip(log_names, log_values)}

            wandb.log(log_dict)

            # else:
            #     logger.info("No references given for %s -> no evaluation.",
            #                 data_set_name)

            if save_attention:
                if attention_scores:
                    attention_name = "{}.{}.att".format(data_set_name, step)
                    attention_path = os.path.join(model_dir, attention_name)
                    logger.info("Saving attention plots. This might take a while..")
                    store_attention_plots(attentions=attention_scores,
                                          targets=hypotheses_raw,
                                          sources=data_set.src,
                                          indices=range(len(hypotheses)),
                                          output_prefix=attention_path)
                    logger.info("Attention plots saved to: %s", attention_path)
                else:
                    logger.warning("Attention scores could not be saved. "
                                   "Note that attention scores are not available "
                                   "when using beam search. "
                                   "Set beam_size to 1 for greedy decoding.")

            if output_path is not None:
                output_path_set = "{}.{}".format(output_path, data_set_name)
                with open(output_path_set, mode="w", encoding="utf-8") as out_file:
                    for hyp in hypotheses:
                        out_file.write(hyp + "\n")
                logger.info("Translations saved to: %s", output_path_set)
        test_duration = time.time() - test_start_time
        logger.info(f"Test duration was {test_duration:.2f}")


def translate(cfg_file: str,
              ckpt: str,
              output_path: str = None,
              batch_class: Batch = Batch) -> None:
    """
    Interactive translation function.
    Loads model from checkpoint and translates either the stdin input or
    asks for input to translate interactively.
    The input has to be pre-processed according to the data that the model
    was trained on, i.e. tokenized or split into subwords.
    Translations are printed to stdout.

    :param cfg_file: path to configuration file
    :param ckpt: path to checkpoint to load
    :param output_path: path to output file
    :param batch_class: class type of batch
    """

    def _load_line_as_data(line):
        """ Create a dataset from one line via a temporary file. """
        # write src input to temporary file
        tmp_name = "tmp"
        tmp_suffix = ".src"
        tmp_filename = tmp_name + tmp_suffix
        with open(tmp_filename, "w") as tmp_file:
            tmp_file.write("{}\n".format(line))

        test_data = MonoDataset(path=tmp_name, ext=tmp_suffix,
                                field=src_field)

        # remove temporary file
        if os.path.exists(tmp_filename):
            os.remove(tmp_filename)

        return test_data

    def _translate_data(test_data):
        # todo modernize data
        """ Translates given dataset, using parameters from outer scope. """
        # pylint: disable=unused-variable
        score, loss, ppl, sources, sources_raw, references, hypotheses, \
        hypotheses_raw, attention_scores, utility, utility_per_sentence, expected_utility_mean = validate_on_data(
            model, data=test_data, batch_size=batch_size,
            batch_class=batch_class, batch_type=batch_type, level=level,
            max_output_length=max_output_length, eval_metric="",
            use_cuda=use_cuda, compute_loss=False, beam_size=beam_size,
            beam_alpha=beam_alpha, postprocess=postprocess,
            bpe_type=bpe_type, sacrebleu_dict=sacrebleu, n_gpu=n_gpu, sample=sample, mbr=mbr,
            mbr_type=mbr_type,
            small_test_run=small_test_run, num_samples=num_samples)
        return hypotheses

    cfg = load_config(cfg_file)
    model_dir = cfg["training"]["model_dir"]

    _ = make_logger(model_dir, mode="translate")
    # version string returned

    # when checkpoint is not specified, take oldest from model dir
    if ckpt is None:
        ckpt = get_latest_checkpoint(model_dir)

    # read vocabs
    src_vocab_file = cfg["data"].get("src_vocab", model_dir + "/src_vocab.txt")
    trg_vocab_file = cfg["data"].get("trg_vocab", model_dir + "/trg_vocab.txt")
    src_vocab = Vocabulary(file=src_vocab_file)
    trg_vocab = Vocabulary(file=trg_vocab_file)

    data_cfg = cfg["data"]
    level = data_cfg["level"]
    lowercase = data_cfg["lowercase"]

    tok_fun = lambda s: list(s) if level == "char" else s.split()

    src_field = Field(init_token=None, eos_token=EOS_TOKEN,
                      pad_token=PAD_TOKEN, tokenize=tok_fun,
                      batch_first=True, lower=lowercase,
                      unk_token=UNK_TOKEN,
                      include_lengths=True)
    src_field.vocab = src_vocab

    # parse test args
    batch_size, batch_type, use_cuda, device, n_gpu, level, _, \
    max_output_length, beam_size, beam_alpha, postprocess, \
    bpe_type, sacrebleu, _, _, sample, mbr, mbr_type, small_test_run, \
    num_samples, utility_type, save_utility_per_sentence, _, _, _, _ = parse_test_args(cfg, mode="translate")

    # load model state from disk
    model_checkpoint = load_checkpoint(ckpt, use_cuda=use_cuda)

    # build model and load parameters into it
    model = build_model(cfg["model"], src_vocab=src_vocab, trg_vocab=trg_vocab)
    model.load_state_dict(model_checkpoint["model_state"])
    model.device = device
    if use_cuda:
        model.to(device)

    if not sys.stdin.isatty():
        # input file given
        test_data = MonoDataset(path=sys.stdin, ext="", field=src_field)
        hypotheses = _translate_data(test_data)

        if output_path is not None:
            # write to outputfile if given
            output_path_set = "{}".format(output_path)
            with open(output_path_set, mode="w", encoding="utf-8") as out_file:
                for hyp in hypotheses:
                    out_file.write(hyp + "\n")
            logger.info("Translations saved to: %s.", output_path_set)
        else:
            # print to stdout
            for hyp in hypotheses:
                print(hyp)

    else:
        # enter interactive mode
        batch_size = 1
        batch_type = "sentence"
        while True:
            try:
                src_input = input("\nPlease enter a source sentence "
                                  "(pre-processed): \n")
                if not src_input.strip():
                    break

                # every line has to be made into dataset
                test_data = _load_line_as_data(line=src_input)

                hypotheses = _translate_data(test_data)
                print("JoeyNMT: {}".format(hypotheses[0]))

            except (KeyboardInterrupt, EOFError):
                print("\nBye.")
                break


from functools import partial


def get_sacre_score_fn(sacre_metric):
    return partial(sacre_score, sacre_metric=sacre_metric)


def sacre_score(hyp, ref, sacre_metric=None):
    return sacre_metric.sentence_score(hyp, [ref]).score


def get_utility_fn(utility_type):
    if utility_type == "editdistance":
        utility_fn = editdistance.eval
    elif utility_type == "beer":
        from mbr_nmt.utility import parse_utility
        utility_fn = parse_utility('beer', lang='en')
    elif utility_type == "meteor":
        from mbr_nmt.utility import parse_utility
        utility_fn = parse_utility('meteor', lang='en')
    else:
        if sacrebleu.__version__ == '2.0.0':

            if utility_type == "bleu":
                bleu = BLEU(smooth_method='exp',
                            effective_order=True)
                utility_fn = get_sacre_score_fn(bleu)  # lambda hyp, ref: bleu.sentence_score(hyp, [ref]).score
            elif utility_type in ["chrf", "chrf++"]:
                word_order = 2 if utility_type == "chrf++" else 0
                chrf = CHRF(word_order=word_order, )
                utility_fn = get_sacre_score_fn(chrf)

        else:
            if utility_type == "bleu":
                utility_fn = lambda hyp, ref: sacrebleu.sentence_bleu(hyp, [ref]).score
            elif utility_type == "chrf":
                utility_fn = lambda hyp, ref: sacrebleu.sentence_chrf(hyp, [ref]).score
            elif utility_type == "chrf++":
                raise Exception("Cannot use chrf++ without sacrebleu of verion 2.0.0")
    return utility_fn


def get_utility_of_samples(utility_type, sample_1, sample_2, reduce_type="mean", save_utility_per_sentence=False):
    """
    Compute mean (by batch) or total edit distance utility between two samples shaped as BxL.
    """
    utility_fn = get_utility_fn(utility_type)
    # iterate over batches in sample
    total_utility = [utility_fn(batch_1, batch_2) for batch_1, batch_2 in zip(sample_1, sample_2)]
    if reduce_type == "mean":
        # compute mean sample utility
        reduced_utility = sum(total_utility) / len(sample_1)
    elif reduce_type == "sum":
        reduced_utility = sum(total_utility)
    # if we will not be saving utility, do not return it.
    if save_utility_per_sentence == False:
        total_utility = None
    return reduced_utility, total_utility


def prepare_encoded_batch_for_sampling(encoded_batch, num_samples):
    try:
        encoded_batch[0] = encoded_batch[0].repeat(num_samples, 1, 1)
    except:
        pass
    try:
        encoded_batch[1] = encoded_batch[1].repeat(num_samples, 1, 1)
    except:
        pass
    return encoded_batch


def mbr_decoding(model, batch, max_output_length=100, num_samples=10, mbr_type="editdistance",
                 utility_type="editdistance", utility_fn=None, return_types=("h",),
                 need_grad=False, compute_log_probs=False, encoded_batch=None, small_test_run=False,
                 rank=None, world_size=1, samples_raw=None):
    set_of_possible_returns = {"h", "samples", "samples_raw", "utilities", "log_probabilities",
                               "mean_batch_expected_uility_h", "batch"}
    assert len(set(return_types) - set_of_possible_returns) == 0, f"You have specified wrong return types. " \
                                                                  f"Make sure it's one (or more) out of" \
                                                                  f" {set_of_possible_returns}"
    # sample S samples of translation y|x  using  run_batch

    # repeat along batch dimension num_samples times to simulate sampling
    batch_size = batch.src.shape[0]
    batch = repeat_batch(batch, num_samples)
    if samples_raw is None:
        if encoded_batch is not None and encoded_batch[0] is not None:
            encoded_batch = prepare_encoded_batch_for_sampling(encoded_batch, num_samples)
        else:
            encoded_batch = None
        # [B*SxL, B*S]
        samples_raw, log_probs = run_batch(model, batch, max_output_length=max_output_length, beam_size=1,
                                           beam_alpha=1,
                                           sample=True, need_grad=need_grad, compute_log_probs=compute_log_probs,
                                           encoded_batch=encoded_batch)
    else:
        assert samples_raw.shape[0] == batch_size * num_samples, \
            f"Something is wrong with sample shape, it must be [B*S, L]," \
            f" it is currently [{samples_raw.shape[0]}, {samples_raw.shape[1]}]," \
            f" while B is {batch_size} and S is {num_samples}"
        log_probs = None
        # if rank is not None:
    # logger.info(f"got samples rank {rank}")

    # [BxSxL] transpose is needed to make sure that it's actually split by batches
    samples = samples_raw.reshape(num_samples, batch_size, -1).transpose((1, 0, 2))
    if "log_probabilities" in return_types and compute_log_probs:
        # [BxS]
        log_probs = log_probs.reshape(batch_size, num_samples)

    if mbr_type == "editdistance":
        # define utility function
        if utility_fn is None:
            utility_fn = get_utility_fn(utility_type)

        # transform indices to str [BxS] if len(sample)!=0 else "the"
        decoded_samples = [[bpe_postprocess(' '.join(sample) + '<\s>') for sample in
                            model.trg_vocab.arrays_to_sentences(arrays=batch, cut_at_eos=True)] for batch in samples]

        # initialise utility matrix [BxSxS]
        # if symmetrical utility, do this
        if utility_type == "editdistance":
            U = torch.zeros([batch_size, num_samples, num_samples], dtype=torch.float) + 1e-10
            # combination of all samples (since utility is symmetrical, we need only AB and not BA samples) [BxC]
            batch_combinations_of_samples = [itertools.combinations(batch_samples, r=2) for batch_samples in
                                             decoded_samples]
            # computing utility of the samples [BxC]
            utilities = torch.tensor([list(itertools.starmap(utility_fn, combinations_of_samples)) for
                                      combinations_of_samples in batch_combinations_of_samples], dtype=torch.float)

            # lower triangular and upper indices of the matrix, below the diagonal, since distance(A,A) = 0
            tril_indices = torch.tril_indices(row=num_samples, col=num_samples, offset=-1)
            triu_indices = torch.triu_indices(row=num_samples, col=num_samples, offset=1)
            # setting the values to the matrix [BxSxS]
            U[:, tril_indices[0], tril_indices[1]] = utilities
            U[:, triu_indices[0], triu_indices[1]] = utilities
        else:
            # permutations of all samples [Bx(S^2)]
            batch_combinations_of_samples = [list(itertools.product(batch_samples, repeat=2)) for batch_samples in
                                             decoded_samples]

            # # computing utility of the samples [Bx(S^2)]
            # [B*S^2]
            # multiprocessing sacre utilities
            if False and utility_type != "beer":
                chunked_batches_and_utility_fns = [(chunked_batches[i], utility_fn[i]) for i in
                                                   range(len(chunked_batches))]
                with mp.Pool(cpus) as p:
                    utilities = p.map(eval_utility_chunked_batch, chunked_batches_and_utility_fns)
                utilities = torch.tensor([u for sublist in utilities for u in sublist], dtype=torch.float)

            elif type(utility_fn) != list:

                utilities = torch.tensor([list(itertools.starmap(utility_fn, combinations_of_samples)) for
                                          combinations_of_samples in batch_combinations_of_samples],
                                         dtype=torch.float)
            # multithreading
            elif type(utility_fn) == list:
                # number of threads is number of cpus per gpu
                cpus = mp.cpu_count() // world_size
                # number of examples per thread
                num_per_cpu = int(np.ceil(len(batch_combinations_of_samples) / cpus))
                # chunk the batch for each thread and combine with specififc utility process
                chunked_batches = [batch_combinations_of_samples[i:i + num_per_cpu]
                                   for i in range(0, len(batch_combinations_of_samples), num_per_cpu)]

                # add the utility fn next to the batches
                chunked_batches_and_utility_fns = [(chunked_batches[i], utility_fn[i]) for i in
                                                   range(len(chunked_batches))]

                # multithread
                with ThreadPool(cpus) as p:
                    utilities = p.map(eval_utility_chunked_batch, chunked_batches_and_utility_fns)
                # flatten the list and wrap in a tensor
                utilities = torch.tensor([u for sublist in utilities for u in sublist], dtype=torch.float)

            # setting the values to the matrix [BxSxS]
            U = utilities.reshape(batch_size, num_samples, num_samples)
        # if rank is not None:
        #     logger.info(f"computed utility rank {rank}")
        # compute utility per candidate [BxS]
        expected_uility = torch.mean(U, dim=-1)
        # get argmin_c or argmax_c of sum^S u(samples, c) (min because we want less edit distance) [B]
        if utility_type == "editdistance":
            best_idx = torch.argmin(expected_uility, dim=-1)
        else:
            best_idx = torch.argmax(expected_uility, dim=-1)
        # return things that you want to return
        return_dict = {}
        if "h" in return_types:
            # get prediction with best samples
            prediction = samples[np.arange(batch_size), best_idx.numpy()]
            return_dict['h'] = prediction
        if "samples" in return_types:
            return_dict['samples'] = samples
        if "samples_raw" in return_types:
            return_dict['samples_raw'] = samples_raw
        if "utilities" in return_types:
            best_idx = best_idx.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, num_samples)
            # [BxS]
            u_h = U.gather(1, best_idx).squeeze(1)
            # print("u_h", u_h)
            u_h = u_h + 1e-10
            return_dict['utilities'] = u_h
        if "log_probabilities" in return_types:
            return_dict['log_probabilities'] = log_probs
        if "batch" in return_types:
            return_dict['batch'] = batch
        if "mean_batch_expected_uility_h" in return_types:
            # logger.info(f"best_idx {best_idx} expected_utility shape {expected_uility.shape}")
            return_dict['mean_batch_expected_uility_h'] = torch.mean(expected_uility[:, best_idx].detach())
        return return_dict



# eval utility of a single batch
def eval_utility_batch(batch, utility_fn):
    utility = [utility_fn(*combination_of_samples) for combination_of_samples in batch]
    return utility


def eval_utility_chunked_batch(batches_and_fn):
    batches, utility_fn = batches_and_fn
    print(len(batches), flush=True)

    utility = [list(itertools.starmap(utility_fn, combinations_of_samples)) for
               combinations_of_samples in batches]
    return utility



if __name__ == "__main__":
    pass
