# coding: utf-8
"""
Module to implement training loss
"""

import torch
from math import log

from torch import nn, Tensor
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_sequence

from joeynmt.prediction import mbr_decoding
from joeynmt.helpers import repeat_batch
from joeynmt.batch import Batch


class XentLoss(nn.Module):
    """
    Cross-Entropy Loss with optional label smoothing
    """

    def __init__(self, pad_index: int, smoothing: float = 0.0, utility_alpha=1, num_samples=1,
                 max_output_length=100, mean_baseline=False, vimco_baseline=False, dynamic_max_sample=False,
                 sampling_max_buffer=5):
        super().__init__()
        self.smoothing = smoothing
        self.pad_index = pad_index
        if self.smoothing <= 0.0:
            # standard xent loss
            self.criterion = nn.NLLLoss(ignore_index=self.pad_index,
                                        reduction='sum')
        else:  # reguralizing
            # custom label-smoothed loss, computed with KL divergence loss
            self.criterion = nn.KLDivLoss(reduction='sum')

        self.no_reduce_criterion = nn.NLLLoss(ignore_index=self.pad_index,
                                              reduction='none')
        # regularisation strength
        self.utility_alpha = utility_alpha
        # number of samples to compute the utility reg
        self.num_samples = num_samples
        # maximum sentece length in a sample
        self.max_output_length = max_output_length
        # whether to change the max depending on the batch size
        self.dynamic_max_sample = dynamic_max_sample
        self.sampling_max_buffer = sampling_max_buffer
        # which variance reduction baselines to use for REINFORCE
        self.mean_baseline = mean_baseline
        self.vimco_baseline = vimco_baseline
        # running average container for the mean baseline
        self._utility_running_average = 0
        self._utility_step = 0
        # initialise utility fn once
        self._utility_fn = None
        self._world_size = 1

    @staticmethod
    def preprocess_samples_for_batch(samples, bos, eos, pad):

        def tensor_of_shape_x_and_all_elements_are_y(x, y):
            tensor_xy = torch.empty(x)
            tensor_xy[:] = y
            return tensor_xy.view(x, 1)

        # print(samples.shape, "is samples shape")
        # print(samples)

        lengths = tensor_of_shape_x_and_all_elements_are_y(samples.shape[0], samples.shape[1]).to(torch.int32)
        # appending bos and and eos
        bos_tensor = tensor_of_shape_x_and_all_elements_are_y(samples.shape[0], bos)
        eos_tensor = tensor_of_shape_x_and_all_elements_are_y(samples.shape[0], eos)
        # print(bos_tensor.shape, samples.shape)
        bos_samples = torch.cat((bos_tensor, torch.tensor(samples)), dim=1)
        # print(bos_samples.shape)
        bos_samples_eos = torch.cat((bos_samples, eos_tensor), dim=1)
        # print("shape after cating", bos_samples_eos.shape)
        # padding
        padded_samples = pad_sequence(bos_samples_eos, batch_first=True, padding_value=pad).to(torch.long)
        # print(f"shaper after padding, {padded_samples.shape}")
        return padded_samples, lengths

    def vimco_baseline_fn(self, log_uh):
        # get all the utilities in the sample[B]
        total_utility = torch.sum(log_uh, dim=1)
        # substract the jth element at the jth index [B,S]
        sum_min_j = total_utility.unsqueeze(-1) - log_uh
        # get the mean without the jth
        mean_min_j = sum_min_j - log(self.num_samples - 1)
        # baseline is the sum without the jth + mean without the jth
        vimco_baseline = sum_min_j + mean_min_j - log(self.num_samples)
        return vimco_baseline

    def mean_baseline_step(self, log_uh):
        self._utility_step += 1
        self._utility_running_average += (torch.mean(log_uh).item() - self._utility_running_average) \
                                         / self._utility_step

    def dynamic_match_sample_size(self, batch):
        if self.dynamic_max_sample:
            max_output_length = torch.max(batch.trg_length)
            if max_output_length <= 50:
                max_output_length += self.sampling_max_buffer
        else:
            max_output_length = self.max_output_length
        return max_output_length

    def utility_loss(self, model, batch, batch_loss, utility_type='beer', encoded_batch=None, samples_raw=None,
                     log_probs_0=0):
        log_dict = {"nll": batch_loss.item()}
        # reinforce: \delta E = E_{p(y|\theta, x)} [log u(y,h) * \delta log p (y|\theta, x)]
        from joeynmt.prediction import mbr_decoding
        # compute mbr, get utility(samples, h)
        # todo pass encode batch to save computations
        # dynamically adjust max
        max_output_length = self.dynamic_match_sample_size(batch)

        # print("Max output length", max_output_length, "Dynamic?", self.dynamic_max_sample, flush=True)
        with torch.no_grad():
            mbr_dict = mbr_decoding(model, batch, max_output_length=max_output_length,
                                    num_samples=self.num_samples,
                                    mbr_type="editdistance", utility_type=utility_type,
                                    return_types=("utilities", "samples_raw", "batch"),
                                    need_grad=False, compute_log_probs=False,
                                    encoded_batch=encoded_batch, utility_fn=self._utility_fn,
                                    world_size=self._world_size, samples_raw=samples_raw)

            # [BxS], [S*BxL]
            u_h, samples, batch = mbr_dict['utilities'], mbr_dict['samples_raw'], mbr_dict['batch']
        log_dict['samples_raw'] = samples.copy()
        # print(f"samples shape {samples.shape}")
        trg_sampled, trg_length = self.preprocess_samples_for_batch(samples, model.bos_index, model.eos_index,
                                                                    model.pad_index)
        # [S*BxL]
        new_batch = Batch((batch.src, batch.src_length, trg_sampled, trg_length), pad_index=model.pad_index,
                          use_cuda=batch.src.is_cuda, device=batch.src.device)
        # print("batch after repeat batch", batch.src.shape, batch.src_length.shape)
        # print(f"new_batch src {new_batch.src.shape}, trg {new_batch.trg.shape}")
        sample_log_probs, _, _, _ = model(return_type="log_prob",
                                          **{"utility_regularising": False,
                                             **vars(new_batch)})
        sample_log_probs = sample_log_probs.reshape((batch.src.shape[0] // self.num_samples, self.num_samples, -1))
        sample_log_probs_per_sentence = sample_log_probs.sum(-1) - log_probs_0
        # print(f"log probs shapes {sample_log_probs.shape}, {log_probs_0.shape if type(log_probs_0) == torch.tensor else 0}, {sample_log_probs_per_sentence.shape} ")
        log_dict["log_probs_0"] = sample_log_probs_per_sentence.detach()
        # computing utility
        # utility logging (pun intended)
        log_uh = torch.log(u_h).detach()
        log_dict['u_h'] = u_h.detach().clone().mean().numpy()
        log_dict['mean_utility'] = u_h.mean().item()
        log_dict['log_u_h'] = log_uh.mean().item()
        # control variates for utility

        # VIMCO control variate from arxiv.org/pdf/1602.06725.pdf
        # vimco_baseline_j = log (\sum_i^{-j} u_i + mean^{-j}) -logS

        if self.vimco_baseline:
            vimco_baseline = self.vimco_baseline_fn(log_uh)
        else:
            vimco_baseline = 0
        # substract the vimco baseline
        log_uh = log_uh - vimco_baseline
        log_dict['mean_vimco_utility'] = log_uh.mean().item()
        log_dict['mean_vimco'] = vimco_baseline.mean().item()

        # if we use mean control variate, substract the current mean and then update the mean
        if self.mean_baseline:
            # new log utility is  log utility
            mean_baseline = self._utility_running_average
            # update running mean += (utility_sample_mean - running mean)/N
            self.mean_baseline_step(log_uh)
        else:
            mean_baseline = 0
        log_dict['mean_baseline'] = mean_baseline
        # substract the mean baseline
        log_uh = log_uh - mean_baseline
        log_dict['mean_utility_after_baselines'] = log_uh.mean().item()
        # compute mean of U(y,h) * \grad p(y)
        # todo maybe clip sample_log_probs_per_sentence
        # sample_log_probs_per_sentence = torch.clip(sample_log_probs_per_sentence, 0.8, 1.2)
        utility_term = torch.mean(log_uh.to(sample_log_probs_per_sentence.device) * sample_log_probs_per_sentence)
        log_dict['utility_term'] = utility_term.item()

        if utility_type in ["beer", "bleu", "chrf", "chrf++"]:
            utility_alpha = self.utility_alpha * -1
        elif utility_type == "edit_distance":
            utility_alpha = self.utility_alpha
        batch_loss += utility_term * utility_alpha

        # utility_term = utility_term.item()

        return batch_loss, log_dict

    def _smooth_targets(self, targets: Tensor, vocab_size: int):
        """
        Smooth target distribution. All non-reference words get uniform
        probability mass according to "smoothing".

        :param targets: target indices, batch*seq_len
        :param vocab_size: size of the output vocabulary
        :return: smoothed target distributions, batch*seq_len x vocab_size
        """
        # batch*seq_len x vocab_size
        smooth_dist = targets.new_zeros((targets.size(0), vocab_size)).float()
        # fill distribution uniformly with smoothing
        smooth_dist.fill_(self.smoothing / (vocab_size - 2))
        # assign true label the probability of 1-smoothing ("confidence")
        smooth_dist.scatter_(1, targets.unsqueeze(1).data, 1.0 - self.smoothing)
        # give padding probability of 0 everywhere
        smooth_dist[:, self.pad_index] = 0
        # masking out padding area (sum of probabilities for padding area = 0)
        padding_positions = torch.nonzero(targets.data == self.pad_index,
                                          as_tuple=False)
        # pylint: disable=len-as-condition
        if len(padding_positions) > 0:
            smooth_dist.index_fill_(0, padding_positions.squeeze(), 0.0)
        return Variable(smooth_dist, requires_grad=False)

        # pylint: disable=arguments-differ

    def forward(self, log_probs, targets, reduce=True):
        """
        Compute the cross-entropy between logits and targets.

        If label smoothing is used, target distributions are not one-hot, but
        "1-smoothing" for the correct target token and the rest of the
        probability mass is uniformly spread across the other tokens.

        :param log_probs: log probabilities as predicted by model
        :param targets: target indices
        :return:
        """
        if self.smoothing > 0:
            targets = self._smooth_targets(
                targets=targets.contiguous().view(-1),
                vocab_size=log_probs.size(-1))
            # targets: distributions with batch*seq_len x vocab_size
            assert log_probs.contiguous().view(-1, log_probs.size(-1)).shape \
                   == targets.shape
        else:
            # targets: indices with batch*seq_len
            targets = targets.contiguous().view(-1)
        if reduce:
            loss = self.criterion(
                log_probs.contiguous().view(-1, log_probs.size(-1)), targets)
        else:
            loss = self.no_reduce_criterion(log_probs.contiguous().view(-1, log_probs.size(-1)), targets)
        return loss
