import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import pyro
import torch
import numpy as np

from pyro.infer.autoguide import AutoDiagonalNormal

import data_generation
from BLR import BayesianRegression


def get_log_likelihood(posterior, model, observations, nchains, ndraws):
    # this is MCMC version from https://github.com/arviz-devs/arviz/blob/f4a00787e2afd031e09660c19ab1d5d6baf4a240/arviz/data/io_pyro.py#L97
    # todo fix to guide version
    data = {}
    samples = posterior.get_samples(group_by_chain=False)
    predictive = pyro.infer.Predictive(model, samples)
    vectorized_trace = predictive.get_vectorized_trace()
    for obs_name in observations.keys():
        obs_site = vectorized_trace.nodes[obs_name]
        log_like = obs_site["fn"].log_prob(obs_site["value"]).detach().cpu().numpy()
        shape = (nchains, ndraws) + log_like.shape[1:]
        data[obs_name] = np.reshape(log_like, shape)


def loss_fn(true, predict):
    return 1


def transform_loss_to_utility(loss, method="exp", gamma=1):
    return torch.exp(-loss * gamma)


def get_predictive_posterior_samples(model, guide, svi=True, num_posterior_samples=200,
                                     names_of_sites_to_return=['obs']):
    """
    Args:
        svi: bool, whether using guide or mcmc
        num_posterior_samples: int, number of posterior samples to generate
        data_len: int, number of data samples for each posterior sample
    Returns: dict, with  "samples": torch.tensor of shape (num_posterior_samples, data_len)

    """
    # with torch.no_grad():
    assert guide is not None, "Please, first train the guide before trying to generate samples from it."
    print(f"Generating {num_posterior_samples} Predictive Posterior Samples using SVI")

    predictive = pyro.infer.Predictive(model, posterior_samples=None, guide=guide,
                                       num_samples=num_posterior_samples, return_sites=names_of_sites_to_return)
    predictive_samples = {k: v.detach().cpu()
                          for k, v in predictive(None).items()}
    return predictive_samples


def utility_regulariser(model, guide, train_X, train_Y, x_target, y_target=None, S=5, M=10, loss_type="SE"):
    # todo make classification problem : seems simpler

    """
    FOR NOW ITS SQUARED LOSS
    Types of utility:
    1. Irrespective of y: \int q_{\lambda}(\theta) \log \int p(y \mid \theta, \mathcal{D}) u(y, h) d y d \theta
                1. Sample S \theta from q(\theta).
                2. Sample N predictive ys based on p(y|x,\theta).
                3. Estimate h: closed form for now
                4. Compute u(y,h)

    Args:
        guide:
        train_X:
        train_Y:
        x_target:
        y_target:

    Returns:

    """
    assert loss_type in ['SE'], "Please, specify a valid loss."
    assert len(train_X.shape) == 2, f"Make sure train_X is of the correct shape: (N,d)." \
                                    f" Currently it's shaped {train_X.shape}"
    args = train_X[:, 0].unsqueeze(1)
    N = args.shape[0]
    loss = 0.0
    predictive_samples = torch.empty(S, M, N)
    predictive = pyro.infer.Predictive(model, posterior_samples=None, guide=guide,
                                       num_samples=M, return_sites=['obs'])
    # Compute predictive samples
    for s in range(S):
        predictive_samples[s] = predictive(args)['obs']
        print(predictive_samples[s].shape)

    print("Final shape is", predictive_samples.shape)
    # Compute decision.
    if loss_type == "SE":
        # The bayes estimator for this loss function is the mean of predictive posterior samples.
        # In code of tkusmierczyk in LCVI directory, he uses mean with respect to both theta and predictive
        h = torch.mean(predictive_samples.view((S*M, N)), dim=0)
        print("H", h.shape)

    # Compute loss
    if loss_type == "SE":
        loss_fn = torch.nn.MSELoss(reduction='mean')
        loss = loss_fn(predictive_samples, h)

    return loss


"""def squared_loss_optimal_h(ys): 
    return ys.mean(0) #allows gradients through     
    
    def squared_loss(h, y):
    return (y-h)**2"""


def run_train():
    train_X, train_Y, target_X, target_Y, test_X, test_Y = data_generation.get_data(N=1000, sine=False,
                                                                                    categories=True,
                                                                                    noise_variation=0.4,
                                                                                    noise_variation_2=0.99, plot=False,
                                                                                    seed=42)
    train(train_X, train_Y)


def train(train_X, train_Y, num_iters=1000, target_X=None, target_Y=None, test_X=None, test_Y=None, lr=0.01, seed=42,
          test=True):
    # PREPARATIONS for training
    # clear param store and seed
    pyro.clear_param_store()
    if seed is not None:
        torch.manual_seed(seed)
    if test:
        num_iters = 10
    # initialise model and guide
    model = BayesianRegression(1, 1)
    guide = AutoDiagonalNormal(model)
    print(train_X[:, 0].unsqueeze_(1).shape, train_Y.shape)
    model_and_guide_args = (train_X[:, 0].unsqueeze_(1), train_Y)
    # populate param_store
    guide(*model_and_guide_args)

    # define optimizer and loss function
    optimizer = torch.optim.Adam(guide.parameters(), **{"lr": lr, "betas": (0.90, 0.999)})
    loss_fn = pyro.infer.Trace_ELBO().differentiable_loss

    # compute loss
    elbo = []
    for i in range(num_iters + 1):
        loss = loss_fn(model, guide,
                       *model_and_guide_args) + utility_regulariser(model, guide, train_X, train_Y, target_X, target_Y)
        loss.backward()
        # take a step and zero the parameter gradients
        optimizer.step()
        optimizer.zero_grad()

        iter_loss = loss.item() / train_X.shape[0]
        elbo.append(iter_loss)
        if i % (1 if test else 100) == 0:
            print(f"[{i :4d}/{num_iters}] ELBO: {iter_loss:.4f}")
    print(f"Final ELBO is {elbo[-1]:.2f}")

    # todo do heldout log likelihood on , test_X, test_Y
    if test_X and test_Y is not None:
        test_args = (test_X, test_Y)
        with torch.no_grad():
            loss = loss_fn(model, guide, *test_args)
        print(f"Test ELBO is {loss/len(test_Y)}")


def main():
    run_train()


if __name__ == '__main__':
    main()
