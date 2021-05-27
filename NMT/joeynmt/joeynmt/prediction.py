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
from torchtext.legacy.data import Field
from torch.utils.data import Dataset, Subset
# from torch.utils.data.distributed import DistributedSampler

from joeynmt.helpers import bpe_postprocess, load_config, make_logger, \
    get_latest_checkpoint, load_checkpoint, store_attention_plots
from joeynmt.metrics import bleu, chrf, token_accuracy, sequence_accuracy
from joeynmt.model import build_model, Model, _DataParallel
from joeynmt.search import run_batch
from joeynmt.batch import Batch
from joeynmt.data import load_data, make_dataloader, MonoDataset, DistributedEvalSampler
from joeynmt.constants import UNK_TOKEN, PAD_TOKEN, EOS_TOKEN
from joeynmt.vocabulary import Vocabulary

# import os
# ROOT_DIR = os.path.abspath(os.curdir)
# print(ROOT_DIR)
# import ..mbr_nmt.utility as utility


logger = logging.getLogger(__name__)


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
                     sacrebleu: dict = None,
                     sample: bool = False,
                     mbr: bool = False,
                     mbr_type: str = 'editdistance',
                     small_test_run: bool = False,
                     num_samples: int = 10,
                     save_utility_per_sentence: bool = False,
                     utility_type=None,
                     rank=0,
                     world_size=1) \
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
    :param sacrebleu: sacrebleu options
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
    assert batch_size >= n_gpu, "batch_size must be bigger than n_gpu."
    if sacrebleu is None:  # assign default value
        sacrebleu = {"remove_whitespace": True, "tokenize": "13a"}
    if batch_size > 1000 and batch_type == "sentence":
        logger.warning(
            "WARNING: Are you sure you meant to work on huge batches like "
            "this? 'batch_size' is > 1000 for sentence-batching. "
            "Consider decreasing it or switching to"
            " 'eval_batch_type: token'.")

    # if small test run, use subset of the data that is one batch
    if small_test_run:
        SMALL_DATA_SIZE = 5

        val_subset_data, _ = torch.utils.data.random_split(data,
                                                           [SMALL_DATA_SIZE, len(data) - SMALL_DATA_SIZE],
                                                           torch.Generator().manual_seed(42))
        val_subset_data.src_vocab = data.src_vocab
        val_subset_data.trg_vocab = data.trg_vocab
        data = val_subset_data
        logger.info(f"The lenght of validation data for small test run is {len(data)}")
        # batch_size = len(data)
    model.ddp = getattr(model, "ddp", False)
    if  model.ddp:
        ddp_sampler = DistributedEvalSampler(data, num_replicas=world_size, rank=rank)
        ddp_indices = ddp_sampler.indices
    else:
        ddp_sampler = None

    valid_dataloader = make_dataloader(
        dataset=data, batch_size=batch_size,
        shuffle=False, train=False, sampler=ddp_sampler)
    valid_sources_raw = [src for src, _, trg, _ in data]
    pad_index = model.src_vocab.stoi[PAD_TOKEN]
    # disable dropout
    model.eval()
    # don't track gradients during validation
    with torch.no_grad():
        all_outputs = []
        valid_attention_scores = []
        total_loss = 0
        total_ntokens = 0
        total_nseqs = 0
        # gather subset of data
        for valid_batch in valid_dataloader:
            # run as during training to get validation loss (e.g. xent)

            batch = batch_class(valid_batch, pad_index, use_cuda=use_cuda, device=model.device)
            # sort batch now by src length and keep track of order
            sort_reverse_index = batch.sort_by_src_length()

            # run as during training with teacher forcing
            if compute_loss and batch.trg is not None:
                batch_loss, _, _, _ = model(return_type="loss", **{"batch": batch, **vars(batch)})
                if n_gpu > 1 and not model.ddp:
                    batch_loss = batch_loss.mean()  # average on multi-gpu
                total_loss += batch_loss
                total_ntokens += batch.ntokens
                total_nseqs += batch.nseqs

            # run as during inference to produce translations
            if not mbr:
                output, attention_scores = run_batch(
                    model=model, batch=batch, beam_size=beam_size,
                    beam_alpha=beam_alpha, max_output_length=max_output_length, sample=sample)
            else:
                output = mbr_decoding(model=model, batch=batch, max_output_length=max_output_length,
                                      compute_log_probs=False, need_grad=False, return_types=["h"],
                                      num_samples=num_samples, mbr_type=mbr_type)[0]
                attention_scores = None

            # sort outputs back to original order
            all_outputs.extend(output[sort_reverse_index])

            valid_attention_scores.extend(
                attention_scores[sort_reverse_index]
                if attention_scores is not None and type(attention_scores) != int else [])

        # get only the data on the current gpu
        if model.ddp:
            data = Subset(data, indices=ddp_indices)
        assert len(all_outputs) == len(data)

        if compute_loss and total_ntokens > 0:
            # total validation loss
            valid_loss = total_loss
            # exponent of token-level negative log prob
            valid_ppl = torch.exp(total_loss / total_ntokens)
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
                    tokenize=sacrebleu["tokenize"])
            elif eval_metric.lower() == 'chrf':
                current_valid_score = chrf(valid_hypotheses, valid_references,
                                           remove_whitespace=sacrebleu["remove_whitespace"])
            elif eval_metric.lower() == 'token_accuracy':
                current_valid_score = token_accuracy(  # supply List[List[str]]
                    # todo remove data.trg
                    list(decoded_valid), list(data.trg))
            elif eval_metric.lower() == 'sequence_accuracy':
                current_valid_score = sequence_accuracy(
                    valid_hypotheses, valid_references)
            elif eval_metric.lower() == 'meteor':
                current_valid_score = meteor_utility(
                    valid_hypotheses, valid_references)
            # compute utility
            if utility_type is not None:

                if utility_type == "editdistance":
                    utility = edit_distance_utility(valid_hypotheses, valid_references, reduce_type="sum")
        else:
            current_valid_score = -1
            utility = 0

        # saves utility per sentence. Not implemented yet
        # if save_utility_per_sentence:
        #     get_utility_per_sentence(eval_metric.lower(), valid_hypotheses, valid_references)

    return current_valid_score, valid_loss, valid_ppl, valid_sources, \
           valid_sources_raw, valid_references, valid_hypotheses, \
           decoded_valid, valid_attention_scores, utility


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

    mbr_text, empty = f"{mbr_type}_MBR_", ""
    decoding_description = f"{'Greedy decoding' if (not sample and not mbr) else f'{mbr_text if mbr else empty}Sample decoding'}" if beam_size < 2 else \
        "Beam search decoding with beam size = {} and alpha = {}". \
            format(beam_size, beam_alpha)

    tokenizer_info = f"[{sacrebleu['tokenize']}]" \
        if eval_metric == "bleu" else ""

    return batch_size, batch_type, use_cuda, device, n_gpu, level, \
           eval_metric, max_output_length, beam_size, beam_alpha, \
           postprocess, bpe_type, sacrebleu, decoding_description, \
           tokenizer_info, sample, mbr, mbr_type, small_test_run, num_samples, utility_type


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

    if len(logger.handlers) == 0:
        _ = make_logger(model_dir, mode="test")  # version string returned

    # when checkpoint is not specified, take latest (best) from model dir
    if ckpt is None:
        ckpt = get_latest_checkpoint(model_dir)
        try:
            step = ckpt.split(model_dir + "/")[1].split(".ckpt")[0]
        except IndexError:
            step = "best"

    # load the data
    if datasets is None:
        _, dev_data, test_data, src_vocab, trg_vocab = load_data(
            data_cfg=cfg["data"], datasets=["dev", "test"])
        data_to_predict = {"dev": dev_data, "test": test_data}
    else:  # avoid to load data again
        data_to_predict = {"dev": datasets["dev"], "test": datasets["test"]}
        src_vocab = datasets["src_vocab"]
        trg_vocab = datasets["trg_vocab"]

    # parse test args
    batch_size, batch_type, use_cuda, device, n_gpu, level, eval_metric, \
    max_output_length, beam_size, beam_alpha, postprocess, \
    bpe_type, sacrebleu, decoding_description, tokenizer_info, \
    sample, mbr, mbr_type, small_test_run, num_samples, utility_type \
        = parse_test_args(cfg, mode="test")

    # load model state from disk
    model_checkpoint = load_checkpoint(ckpt, use_cuda=use_cuda)

    # build model and load parameters into it
    model = build_model(cfg["model"], src_vocab=src_vocab, trg_vocab=trg_vocab)
    model.load_state_dict(model_checkpoint["model_state"])
    model.device = device

    if use_cuda:
        model.to(device)

    # multi-gpu eval
    if n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = _DataParallel(model)

    for data_set_name, data_set in data_to_predict.items():
        if data_set is None:
            continue

        dataset_file = cfg["data"][data_set_name] + "." + cfg["data"]["trg"]
        logger.info("Decoding on %s set (%s)...", data_set_name, dataset_file)

        # pylint: disable=unused-variable
        score, loss, ppl, sources, sources_raw, references, hypotheses, \
        hypotheses_raw, attention_scores, utility = validate_on_data(
            model, data=data_set, batch_size=batch_size,
            batch_class=batch_class, batch_type=batch_type, level=level,
            max_output_length=max_output_length, eval_metric=eval_metric,
            use_cuda=use_cuda, compute_loss=False, beam_size=beam_size,
            beam_alpha=beam_alpha, postprocess=postprocess,
            bpe_type=bpe_type, sacrebleu=sacrebleu, n_gpu=n_gpu, sample=sample, mbr=mbr,
            small_test_run=small_test_run, num_samples=num_samples, mbr_type=mbr_type,
            save_utility_per_sentence=save_utility_per_sentence, utility_type=utility_type)
        # pylint: enable=unused-variable

        # if "trg" in data_set.fields:
        logger.info("%4s %s%s: %6.2f [%s]",
                    data_set_name, eval_metric, tokenizer_info,
                    score, decoding_description)
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
        hypotheses_raw, attention_scores, utility = validate_on_data(
            model, data=test_data, batch_size=batch_size,
            batch_class=batch_class, batch_type=batch_type, level=level,
            max_output_length=max_output_length, eval_metric="",
            use_cuda=use_cuda, compute_loss=False, beam_size=beam_size,
            beam_alpha=beam_alpha, postprocess=postprocess,
            bpe_type=bpe_type, sacrebleu=sacrebleu, n_gpu=n_gpu, sample=sample, mbr=mbr,
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
    num_samples, utility_type = parse_test_args(cfg, mode="translate")

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


def meteor_utility(candidate, sample):
    from mbr_nmt.utility import parse_utility
    utility_fn = parse_utility('meteor', lang='en')
    return utility_fn(hyp=candidate, ref=sample)


def edit_distance_utility(sample_1, sample_2, reduce_type="mean"):
    """
    Compute mean (by batch) or total edit distance utility between two samples shaped as BxL.
    """
    # iterate over batches in sample
    total_utility = sum([editdistance.eval(batch_1, batch_2) for batch_1, batch_2 in zip(sample_1, sample_2)])
    if reduce_type == "mean":
        # compute mean sample utility
        return total_utility / len(sample_1)
    elif reduce_type == "sum":
        return total_utility


def mbr_decoding(model, batch, max_output_length=100, num_samples=10, mbr_type="editdistance", return_types=("h",),
                 need_grad=False, compute_log_probs=False, encoded_batch=None):
    set_of_possible_returns = {"h", "samples", "utilities", "log_probabilities"}
    assert len(set(return_types) - set_of_possible_returns) == 0, f"You have specified wrong return types. " \
                                                                  f"Make sure it's one (or more) out of {set_of_possible_returns}"
    # sample S samples of translation y|x  using  run_batch

    # copy batch num_samples times to sample
    batch_size = batch.src.shape[0]
    batch.src = batch.src.repeat(num_samples, 1)
    batch.src_mask = batch.src_mask.repeat(num_samples, 1, 1)
    batch.trg_mask = batch.trg_mask.repeat(num_samples, 1, 1)
    # logger.info(f" batch device in mbr {batch.src.device}")
    # [B*SxL, B*S]
    samples, log_probs = run_batch(model, batch, max_output_length=max_output_length, beam_size=1, beam_alpha=1,
                                   sample=True, need_grad=need_grad, compute_log_probs=compute_log_probs,
                                   encoded_batch=encoded_batch)

    # [BxSxL] transpose is needed to make sure that it's actually split by batches
    samples = samples.reshape(num_samples, batch_size, -1).transpose((1, 0, 2))
    if "log_probabilities" in return_types:
        # [BxS]
        log_probs = log_probs.reshape(batch_size, num_samples)

    if mbr_type == "editdistance":
        # define utility function
        utility_fn = editdistance.eval

        # transform indices to str
        decoded_samples = [
            [' '.join(sample) for sample in model.trg_vocab.arrays_to_sentences(arrays=batch,
                                                                                cut_at_eos=True)] for batch in samples]
        # initialise utility matrix [BxSxS]
        U = torch.zeros([batch_size, num_samples, num_samples], dtype=torch.float) + 1e-10
        # combination of all samples (since utility is symmetrical, we need only AB and not BA samples) [BxC]
        batch_combinations_of_samples = [itertools.combinations(batch_samples, r=2) for batch_samples in
                                         decoded_samples]
        # computing utility of the samples [BxS]
        utilities = torch.tensor([list(itertools.starmap(utility_fn, combinations_of_samples)) for
                                  combinations_of_samples in batch_combinations_of_samples]).to(torch.float)
        # lower triangular and upper indices of the matrix, below the diagonal, since distance(A,A) = 0
        tril_indices = torch.tril_indices(row=num_samples, col=num_samples, offset=-1)
        triu_indices = torch.triu_indices(row=num_samples, col=num_samples, offset=1)
        # setting the values to the matrix [BxSxS]
        U[:, tril_indices[0], tril_indices[1]] = utilities
        U[:, triu_indices[0], triu_indices[1]] = utilities
        # compute utility per candidate [BxS]
        expected_uility = torch.mean(U, dim=-1)
        # get argmin_c of sum^S u(samples, c) (min because we want less edit distance) [B]
        best_idx = torch.argmin(expected_uility, dim=-1)
        # return things that you want to return
        return_list = []
        if "h" in return_types:
            # get prediction with best samples
            prediction = samples[np.arange(batch_size), best_idx.numpy()]
            return_list.append(prediction)
        if "samples" in return_types:
            return_list.append(samples)
        if "utilities" in return_types:
            best_idx = best_idx.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, num_samples)
            # [BxS]
            u_h = U.gather(1, best_idx).squeeze(1)
            u_h = u_h + 1e-10
            return_list.append(u_h)
        if "log_probabilities" in return_types:
            return_list.append(log_probs)
        return return_list


    else:
        from mbr_nmt import mbr, utility
        utility_fn = utility.parse_utility('meteor', 'en')

        # make it list of batches with list of samples inside: BxSxL
        print(samples[0].shape, samples[1].shape)
        batch_first_samples = np.empty((samples[0].shape[0], len(samples), samples[0].shape[1]))
        print(batch_first_samples.shape)

        for i, sample in enumerate(samples):
            batch_first_samples[:, i, :] = sample

        # print(batch_first_samples)
        # print(batch_first_samples.tolist())
        # each batch is converted to list
        # string_samples = list(map(lambda batch: list(map(lambda sentence: " ".join(list(map(str, sentence))), batch)), batch_first_samples.tolist()))
        # todo fix meteor
        # convert to string in list of lists of lists
        string_samples = batch_first_samples.astype("U10").tolist()
        # join the sentences
        string_samples = list(map(lambda batch: list(map(lambda sample: ' '.join(sample), batch)), string_samples))
        # string_samples = list(map(lambda sample: list(map(lambda sentence: list(map(str, sentence)), sample.tolist())), samples))
        print(samples[0].shape)
        print(string_samples)
        print(len(string_samples), len(string_samples[0]), len(string_samples[0][0]))
        # string_samples = list(map(lambda sample: list(map(lambda batch: ''.join(list(map(str,batch))) , sample.tolist()) ),big_l))
        prediction_idx, prediction = mbr.mbr(string_samples, utility_fn)
        return prediction


def get_utility_per_sentence(utility, hyp_samples, ref_samples):
    # todo
    raise NotImplementedError


if __name__ == "__main__":
    pass
