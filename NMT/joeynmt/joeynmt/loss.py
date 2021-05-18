# coding: utf-8
"""
Module to implement training loss
"""

import torch
from torch import nn, Tensor
from torch.autograd import Variable
from joeynmt.prediction import mbr_decoding


class XentLoss(nn.Module):
    """
    Cross-Entropy Loss with optional label smoothing
    """

    def __init__(self, pad_index: int, smoothing: float = 0.0, utility_alpha=0,
                 num_samples=1, max_output_length=100, mean_baseline=False):
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

        self.utility_alpha = utility_alpha  # set by the TrainManager
        self.num_samples = num_samples  # set by the TrainManager
        self.max_output_length = None  # set by the TrainManager
        self._utility_running_average = 0
        self._utility_step = 0
        self.mean_baseline = mean_baseline
        self.vimco_baseline = True

    def utility_loss(self, model, batch, batch_loss):
        # reinforce: \delta E = E_{p(y|\theta, x)} [log u(y,h) * \delta log p (y|\theta, x)]
        from joeynmt.prediction import mbr_decoding
        # compute mbr, get utility(samples, h)
        # todo pass encode batch to save computations
        u_h, sample_log_probs = mbr_decoding(model, batch, max_output_length=self.max_output_length,
                                             num_samples=self.num_samples,
                                             mbr_type="editdistance",
                                             return_types=("utilities", "log_probabilities"),
                                             need_grad=True, compute_log_probs=True,
                                             encoded_batch=None)
        # get log_u(samples,h) and detach for reinforce
        log_uh = torch.log(u_h).detach()
        print(f"log_uh shape {log_uh.shape}")
        # if we use mean control variate, substract the current mean and then update the mean
        if self.mean_baseline:
            # new log utility is  log utility
            mean_baseline = self._utility_running_average
            self._utility_step += 1
            self._utility_running_average += (torch.mean(
                log_uh).item() - self._utility_running_average) / self._utility_step
            log_uh = updated_log_uh
        else:
            mean_baseline = 0
        if self.vimco_baseline:
            total_utility = torch.sum(log_uh)
            print(f"total_utility shape {total_utility.shape}")
            utility_mean = total_utility - torch.log(log_uh.shape[0])
            sum_min_i = total_utility - log_uh - torch.log(log_uh.shape[0] - 1)
            print(f"sum_min_shape {sum_min_i.shape}")
            vimco_baseline =sum_min_i # utility_mean -
        else:
            vimco_baseline = 0
        log_uh = log_uh - mean_baseline - vimco_baseline
        utility_term = torch.mean(log_uh.to(sample_log_probs.device) * sample_log_probs)
        if torch.isinf(utility_term).any():
            logger.info("INF UTILITY" * 100)
            logger.info(f"log_uh is inf? {torch.isinf(log_uh).any()}")
            logger.info(f"sample_log_probs is inf? {torch.isinf(sample_log_probs).any()}")

        if torch.isinf(batch_loss).any():
            logger.info("INF batch loss")

        batch_loss += utility_term * self.utility_alpha
        utility_term = utility_term.item()
        return batch_loss, utility_term, u_h

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
