import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO, Predictive

import torch
import torch.distributions.constraints as constraints

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from general.utility_functions import moving_average, plot_elbo, summary, predictive_checks
from general.model import Model


class Gaussian(Model):

    def __init__(self, data, fixed_scale=True):
        super().__init__(data)
        self.fixed_scale = fixed_scale
        self.freeze_svi = None
        self.ppvb_guide, self.guide = [None] * 2

    def model(self, data, data_len=None, ppvb_batch_size=1):
        # priors
        mean = pyro.sample('mean', dist.Normal(loc=torch.tensor(0.),
                                               scale=torch.tensor(100.)))
        if self.fixed_scale:
            std = torch.tensor(1.)
        else:
            std = pyro.sample('std', dist.HalfCauchy(scale=torch.tensor(10.)))

        # likelihood
        data_len = self._get_data_len(data_len)
        with pyro.plate('data', data_len):
            pyro.sample('obs', dist.Normal(loc=mean, scale=std),
                        obs=data)

    def guide_fn(self, data, data_len=None, ppvb_batch_size=1):
        # variational parameters
        mean_loc = pyro.param('mean_loc', torch.tensor(0.))
        mean_scale = pyro.param('mean_scale', torch.tensor(100.),
                                constraint=constraints.positive)

        # factorized distribution
        pyro.sample('mean', dist.Normal(loc=mean_loc, scale=mean_scale))
        if not self.fixed_scale:
            std_loc = pyro.param('std_loc', torch.tensor(1.), constraint=constraints.positive)
            std_scale = pyro.param('std_scale', torch.tensor(1.), constraint=constraints.positive)

            precision = pyro.sample('std', dist.LogNormal(std_loc,std_scale))
            # std = pyro.sample("std", dist.Delta(precision**2))

    def train_svi(self, num_iters=1000, lr=0.05,
                  ppvb=False, ppvb_batch_size=10,
                  warm_up_ppvb=False, warm_up_num_steps=1000, freeze_svi=False,
                  plot=True, use_print=True, print_every=500):

        self.freeze_svi = freeze_svi
        # if warming up ppvb, either use pretrained SVI guide or train one
        if warm_up_ppvb and ppvb:
            self._warm_up_ppvb(warm_up_num_steps)
        else:
            print("Clearing parameter store.")
            pyro.clear_param_store()

        if ppvb:
            guide = self.pp_guide
            elbo = self.predictive_posterior_elbo
        else:
            guide = self.guide_fn
            elbo = Trace_ELBO(max_plate_nesting=1)

        # initialising optimisation stuff
        optim = pyro.optim.ClippedAdam({'lr': lr, 'betas': [0.9, 0.99], 'clip_norm': 5.0})
        svi = SVI(self.model, guide, optim, loss=elbo)

        elbo_list = []
        if use_print:
            print(f"Training {'SVI' if not ppvb else 'PPVB'}")
        for i in range(num_iters + 2):
            kwargs = {'ppvb_batch_size': ppvb_batch_size} if ppvb else {}
            elbo = svi.step(self.data, **kwargs)
            elbo_list.append(-elbo)
            if i % print_every == 0 and use_print:
                print(f"Step: {i}, Elbo loss: {moving_average(np.array(elbo_list), 1)[-1] :.2f}")

        if plot:
            plot_elbo(elbo_list[10:], path="results/gaussian_elbo.png")
        if ppvb:
            self.ppvb_guide = guide
        else:
            self.guide = guide
        return guide

    # PPVB
    def pp_guide(self, data, data_len=None, ppvb_batch_size=1):
        initialise_from_svi = True
        # define params of the guide for the posterior
        if self.guide is None:
            svi_guide = self.guide_fn
        else:
            svi_guide = self.guide

        if self.freeze_svi:
            with pyro.poutine.block(hide_types=["param"]):
                svi_guide(data, data_len)
        else:
            svi_guide(data, data_len)

        # params that define the mean
        if initialise_from_svi:
            pred_mean_loc = pyro.param('pred_mean_loc', pyro.get_param_store()['mean_loc'].clone().detach())
            pred_mean_scale = pyro.param('pred_mean_scale', pyro.get_param_store()['mean_scale'].clone().detach(),
                                         constraint=constraints.positive)
            pred_scale_loc = pyro.param('pred_scale_loc', pyro.get_param_store()['std_loc'].clone().detach(),
                                        constraint=constraints.positive)
        else:
            pred_mean_loc = pyro.param('pred_mean_loc', torch.tensor(0.))
            pred_mean_scale = pyro.param('pred_mean_scale', torch.tensor(100.),
                                         constraint=constraints.positive)
            pred_scale_loc = pyro.param('pred_scale_loc', torch.tensor(2.),
                                        constraint=constraints.positive)

        pred_mean = pyro.sample('pred_mean', dist.Normal(pred_mean_loc, pred_mean_scale))
        # sampling global variables
        if self.fixed_scale:
            pred_scale = torch.tensor(1.)
        else:
            # pred_scale = pyro.param('pred_scale', torch.tensor(1), constraint=constraints.positive)
            pred_scale = pyro.sample('pred_scale', dist.LogNormal(pred_scale_loc, torch.tensor(2.)))

        # pred_mean = pyro.sample('pred_mean', dist.Normal(pred_mean_loc, pred_mean_scale))

        # sampling pred data
        with pyro.plate('pred_data', ppvb_batch_size):
            pyro.sample("pred", dist.Normal(pred_mean, pred_scale))
