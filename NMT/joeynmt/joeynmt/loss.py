# coding: utf-8
"""
Module to implement training loss
"""

import torch
from math import log

from torch import nn, Tensor
from torch.autograd import Variable
from joeynmt.prediction import mbr_decoding


class XentLoss(nn.Module):
    """
    Cross-Entropy Loss with optional label smoothing
    """

    def __init__(self, pad_index: int, smoothing: float = 0.0, utility_alpha=1, num_samples=1,
                 max_output_length=100, mean_baseline=False, vimco_baseline=False):
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
        # regularisation strength
        self.utility_alpha = utility_alpha
        # number of samples to compute the utility reg
        self.num_samples = num_samples
        # maximum sentece length in a sample
        self.max_output_length = max_output_length
        # which variance reduction baselines to use for REINFORCE
        self.mean_baseline = mean_baseline
        self.vimco_baseline = vimco_baseline
        # running average container for the mean baseline
        self._utility_running_average = 0
        self._utility_step = 0

    def utility_loss(self, model, batch, batch_loss, utility_type='editdistance'):
        log_dict = {"nll": batch_loss.item()}
        # reinforce: \delta E = E_{p(y|\theta, x)} [log u(y,h) * \delta log p (y|\theta, x)]
        from joeynmt.prediction import mbr_decoding
        # compute mbr, get utility(samples, h)
        # todo pass encode batch to save computations
        u_h, sample_log_probs = mbr_decoding(model, batch, max_output_length=self.max_output_length,
                                             num_samples=self.num_samples,
                                             mbr_type="editdistance", utility_type=utility_type,
                                             return_types=("utilities", "log_probabilities"),
                                             need_grad=True, compute_log_probs=True,
                                             encoded_batch=None)

        # log_uh = torch.log(u_h).detach()
        log_dict['u_h'] = u_h.detach().clone().mean(dim=1).numpy()
        log_dict['mean_utility'] = u_h.mean().item()
        # VIMCO control variate from arxiv.org/pdf/1602.06725.pdf
        # vimco_baseline_j = log (\sum_i^{-j} u_i + mean^{-j}) -logS
        # todo maybe log sum exp + log sub exp !
        # todo look into beer range
        if self.vimco_baseline:
            # get all the utilities in the sample[B]
            total_utility = torch.sum(u_h, dim=1)
            # substract the jth element at the jth index [B,S]
            sum_min_j = total_utility.unsqueeze(-1) - u_h
            # get the mean without the jth
            mean_min_j = sum_min_j - log(self.num_samples - 1)
            # baseline is the sum without the jth + mean without the jth
            vimco_baseline = sum_min_j + mean_min_j - log(self.num_samples)
        else:
            vimco_baseline = 0
        # substract the vimco baseline
        u_h = u_h - vimco_baseline
        log_dict['mean_vimco_utility'] = u_h.mean().item()
        log_dict['mean_vimco'] = vimco_baseline.mean().item()

        # if we use mean control variate, substract the current mean and then update the mean
        if self.mean_baseline:
            # new log utility is  log utility
            mean_baseline = self._utility_running_average
            # update running mean += (utility_sample_mean - running mean)/N
            self._utility_step += 1
            self._utility_running_average += (torch.mean(u_h).item() - self._utility_running_average) \
                                             / self._utility_step
        else:
            mean_baseline = 0
        log_dict['mean_baseline'] = mean_baseline
        # substract the mean baseline
        u_h = u_h - mean_baseline
        log_dict['mean_utility_after_baselines'] = u_h.mean().item()
        # compute mean of U(y,h) * \grad p(y)
        utility_term = torch.mean(u_h.to(sample_log_probs.device) * sample_log_probs)
        log_dict['utility_term'] = utility_term.item()
        # add to the batch loss
        batch_loss += utility_term * self.utility_alpha
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

    def forward(self, log_probs, targets):
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
        loss = self.criterion(
            log_probs.contiguous().view(-1, log_probs.size(-1)), targets)
        return loss
