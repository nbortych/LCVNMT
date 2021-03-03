import pyro
from pyro import poutine
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO, Predictive
from pyro.infer.autoguide import AutoMultivariateNormal, AutoDiagonalNormal
import torch
import torch.distributions.constraints as constraints

import numpy as np

from general.utility_functions import moving_average, plot_elbo, summary, predictive_checks
from general.model import Model


class MultivariateGaussian(Model):

    def __init__(self, data, d=2, reparametrised_covarience=False, guide_type="AutoDiagonalNormal"):
        super().__init__(data)
        self.d = d
        self.reparametrised_covarience = reparametrised_covarience

        possible_guides = ["AutoDiagonalNormal", "AutoMultivariateNormal", "guide_fn"]
        assert guide_type in possible_guides, f"Please, make sure that the guide_type is one of {possible_guides}"
        self.guide_type = guide_type

    def model(self, data, data_len=None, ppvb_batch_size=1):

        if self.reparametrised_covarience:
            # Vector of variances for each of the d variables
            theta = pyro.sample("theta", dist.HalfCauchy(torch.ones(self.d)).to_event(1))

            # Lower cholesky factor of a correlation matrix
            eta = torch.ones(1)  # Implies a uniform distribution over correlation matrices
            L_omega = pyro.sample("L_omega", dist.LKJCorrCholesky(self.d, eta))
            # Lower cholesky factor of the covariance matrix
            L_Omega = torch.mm(torch.diag(theta.sqrt()), L_omega)
            # For inference with batches, one might prefer to use torch.bmm(theta.sqrt().diag_embed(), L_omega)
        else:
            # covariance_vector = pyro.sample("covariance_vector", dist.Normal(torch.zeros(self.d), torch.ones(self.d)).to_event(2))
            covariance = torch.eye(self.d)  # torch.diag(covariance_vector)
        # Vector of expectations
        mu = pyro.sample('mean', dist.Normal(torch.zeros(self.d), torch.ones(self.d)).to_event(1))

        data_len = self._get_data_len(data_len)
        with pyro.plate("data", data_len):
            kwargs = dict(scale_tril=L_Omega) if self.reparametrised_covarience else dict(covariance_matrix=covariance)
            pyro.sample("obs", dist.MultivariateNormal(mu, **kwargs), obs=data)

    def guide_fn(self, data, data_len=None, ppvb_batch_size=1):

        # variational parameters
        mean_loc = pyro.param('mean_loc', torch.zeros(self.d))
        mean_scale = pyro.param('mean_scale', torch.ones(self.d),
                                constraint=constraints.positive)

        pyro.sample('mean', dist.Normal(loc=mean_loc, scale=mean_scale))

        theta_loc = pyro.param('theta_loc', torch.ones([self.d]))
        L_omega_eta = pyro.param('eta', torch.tensor(1.))
        # std_loc = pyro.param('std_loc', torch.tensor(2.), constraint=constraints.positive)

        # Distribution
        pyro.sample("theta", dist.HalfCauchy(theta_loc))
        pyro.sample("L_omega", dist.LKJCorrCholesky(self.d, L_omega_eta))

    def train_svi(self, num_iters=1000, plot=True, ppvb=False, lr=0.05, ppvb_batch_size=10):
        if ppvb:
            guide = self.pp_guide
            elbo = self.predictive_posterior_elbo
        else:
            guide = self._get_guide_fn()
            elbo = Trace_ELBO(max_plate_nesting=1)
        # initialising optimisation stuff

        optim = pyro.optim.ClippedAdam({'lr': lr, 'betas': [0.9, 0.99], 'clip_norm': 5.0})
        svi = SVI(self.model, guide, optim, loss=elbo)
        pyro.clear_param_store()

        elbo_list = []
        print("Training SVI")
        for i in range(num_iters + 2):
            kwargs = {'ppvb_batch_size': ppvb_batch_size} if ppvb else {}
            elbo = svi.step(self.data, **kwargs)
            elbo_list.append(-elbo)
            if i % 500 == 0:
                print(f"Step: {i}, Elbo loss: {moving_average(np.array(elbo_list), 1)[-1] :.2f}")

        if plot:
            plot_elbo(elbo_list[10:])
        if ppvb:
            self.ppvb_guide = guide
        else:
            self.guide = guide
        return guide

    def _get_guide_fn(self):
        guide_options = {"AutoDiagonalNormal": AutoDiagonalNormal(self.model),
                         "AutoMultivariateNormal": AutoMultivariateNormal(self.model),
                         "guide_fn": self.guide_fn}

        guide = guide_options[self.guide_type]
        return guide

    # PPVB
    def pp_guide(self, data, data_len=None, ppvb_batch_size=1):
        # define params of the guide for the posterior
        data_len = self._get_data_len(data_len)
        # define params of the guide for the posterior
        if self.guide is None:
            svi_guide = self._get_guide_fn()
        else:
            svi_guide = self.guide

        # if self.freeze_svi:
        #     with poutine.block(hide_types=["param"]):
        #         svi_guide(data, data_len)
        # else:
        if data is not None:
            svi_guide(data, data_len)

        # params that define the mean
        # pred_mean_loc = pyro.param('pred_mean_loc', torch.zeros(self.d))
        # pred_mean_scale = pyro.param('pred_mean_scale', torch.ones(self.d),
        #                              constraint=constraints.positive)
        #
        # pred_scale_loc = pyro.param('pred_scale_loc', torch.tensor(10),
        #                             constraint=constraints.positive)

        # sampling global variables
        if self.reparametrised_covarience:
            theta = pyro.param("pred_theta", torch.ones(self.d))
            eta = pyro.param('pred_eta', torch.ones(1))
            L_omega = pyro.param('pred_L_omega', torch.randn(self.d, self.d))
            L_omega = torch.tril(L_omega)
            #L_omega = pyro.sample("pred_L_omega", dist.LKJCorrCholesky(self.d, eta))
            # Lower cholesky factor of the covariance matrix
            pred_omega = torch.mm(torch.diag(theta.sqrt()), L_omega)
        else:
            pred_scale = pyro.param('pred_scale', torch.eye(self.d) * 2)
            # , constraint=constraints.positive_definite)  # pyro.sample('pred_scale', dist.LogNormal(pred_scale_loc, torch.tensor(2.)))

        pred_mean = pyro.param('pred_mean', torch.randn(
            self.d) * 2)  # pyro.sample('pred_mean', dist.Normal(pred_mean_loc, pred_mean_scale))

        # sampling pred data
        with pyro.plate('pred_data', ppvb_batch_size):
            kwargs = dict(scale_tril=pred_omega) if self.reparametrised_covarience \
                else dict(covariance_matrix=pred_scale)
            pyro.sample("pred", dist.MultivariateNormal(loc=pred_mean, **kwargs))
