import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import pyro
import torch
import numpy as np
import matplotlib.pyplot as plt

from pyro.infer.autoguide import AutoDiagonalNormal
from pyro.infer import Predictive
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


# todo try polynomial regression
def designmatrix(x, M):
    phi = torch.empty(M, x.shape[0])

    for i in range(M):
        power = torch.pow(x, i + 1).reshape(x.shape[0], 1)
        phi[i] = power

    return phi.T


def transform_loss_to_utility(loss, method="exp", gamma=1):
    return torch.exp(-loss * gamma)


def utility_regulariser(model, guide, train_X, train_Y, x_target, y_target=None, unsupervised_utility=True, S=5, M=10,
                        loss_type="SE", EM=False):
    # todo make classification problem

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

    if unsupervised_utility:
        assert x_target is not None, "Please, pass the unlabeled data."
        data = x_target
    else:
        data = train_X
    # check for dimensionality of data
    assert len(data.shape) == 2, f"Make sure the {'train' if not unsupervised_utility else 'target'}_X" \
                          f" is of the correct shape: (N,d). Currently it's shaped {train_X.shape}"
    args = data[:, 0].unsqueeze(1)
    N = args.shape[0]
    loss = 0.0
    predictive_samples = torch.empty(S, M, N)
    predictive = Predictive(model, posterior_samples=None, guide=guide,
                            num_samples=M, return_sites=['obs'])
    # Compute predictive samples
    for s in range(S):
        predictive_samples[s] = predictive(args)['obs']
        # print(predictive_samples[s].shape)

    # Compute decision.
    if loss_type == "SE":
        # The bayes estimator for this loss function is the mean of predictive posterior samples.
        # In code of tkusmierczyk in LCVI directory, he uses mean with respect to both theta and predictive
        h = torch.mean(predictive_samples.view((S * M, N)), dim=0)
        h = h.unsqueeze(0).expand(S * M, N)

    # If we're using Expectation Maximisation, then sample it again for E step (and then the previous was the M step)
    if EM:
        for s in range(S):
            predictive_samples[s] = predictive(args)['obs']
    # Compute loss
    if loss_type == "SE":
        loss_fn = torch.nn.MSELoss(reduction='sum')
        loss = loss_fn(predictive_samples.view((S * M, N)), h)

    return loss
#
# def categorical_utility_regulariser(model, guide, train_X, train_Y, x_target, y_target=None, unsupervised_utility=True, S=5, M=10,
#                         loss_type="SE", EM=False):
#     # todo make classification problem
#
#     """
#     FOR NOW ITS SQUARED LOSS
#     Types of utility:
#     1. Irrespective of y: \int q_{\lambda}(\theta) \log \int p(y \mid \theta, \mathcal{D}) u(y, h) d y d \theta
#                 1. Sample S \theta from q(\theta).
#                 2. Sample N predictive ys based on p(y|x,\theta).
#                 3. Estimate h: closed form for now
#                 4. Compute u(y,h)
#
#     Args:
#         guide:
#         train_X:
#         train_Y:
#         x_target:
#         y_target:
#
#     Returns:
#
#     """
#     assert loss_type in ['SE'], "Please, specify a valid loss."
#
#     if unsupervised_utility:
#         assert x_target is not None, "Please, pass the unlabeled data."
#         data = x_target
#     else:
#         data = train_X
#     # check for dimensionality of data
#     assert len(data.shape) == 2, f"Make sure the {'train' if not unsupervised_utility else 'target'}_X" \
#                                  f" is of the correct shape: (N,d). Currently it's shaped {train_X.shape}"
#     args = data
#     N = args.shape[0]
#     num_categories = args[:,1].unique()
#     print(num_categories)
#     loss = 0.0
#     predictive_samples = torch.empty(S, M, N)
#     predictive = Predictive(model, posterior_samples=None, guide=guide,
#                             num_samples=M, return_sites=['obs'])
#     # Compute predictive samples
#     for s in range(S):
#         predictive_samples[s] = predictive(args)['obs']
#         # print(predictive_samples[s].shape)
#
#     # Compute decision.
#     if loss_type == "SE":
#         # The bayes estimator for this loss function is the mean of predictive posterior samples.
#         # In code of tkusmierczyk in LCVI directory, he uses mean with respect to both theta and predictive
#         h = torch.mean(predictive_samples.view((S * M, N)), dim=0)
#         h = h.unsqueeze(0).expand(S * M, N)
#
#     # If we're using Expectation Maximisation, then sample it again for E step (and then the previous was the M step)
#     if EM:
#         for s in range(S):
#             predictive_samples[s] = predictive(args)['obs']
#     # Compute loss
#     if loss_type == "SE":
#         loss_fn = torch.nn.MSELoss(reduction='sum')
#         loss = loss_fn(predictive_samples.view((S * M, N)), h)
#
#     return loss

"""def squared_loss_optimal_h(ys): 
    return ys.mean(0) #allows gradients through     
    
    def squared_loss(h, y):
    return (y-h)**2"""


def run_train(sine=True, categories=False, unsupervised_utility=True, num_iters=1000, seed=42):
    train_X, train_Y, target_X, target_Y, test_X, test_Y, params_for_plotting = data_generation.get_data(N=1000,
                                                                                                         sine=sine,
                                                                                                         categories=categories,
                                                                                                         noise_variation=1.1,
                                                                                                         noise_variation_2=0.99,
                                                                                                         plot=False,
                                                                                                         seed=seed)
    model, guide = train(train_X, train_Y, num_iters, target_X, target_Y, test_X, test_Y,
                         unsupervised_utility=unsupervised_utility,
                         seed=seed)

    pred_summary = get_predictive(model, guide, params_for_plotting["X"])
    plot_predictive(pred_summary, params_for_plotting)


def train(train_X, train_Y, num_iters=1000, target_X=None, target_Y=None, test_X=None, test_Y=None, lr=0.01, seed=42,
          test=False, unsupervised_utility=False, categorical_utility= False, utility_calibration=True, EM=False, lazy_regularisation=None):
    # PREPARATIONS for training
    # clear param store and seed
    pyro.clear_param_store()
    if seed is not None:
        torch.manual_seed(seed)
    if test:
        num_iters = 10
    # Shal we use categories for utility
    if not categorical_utility:
        utility_fn = utility_regulariser
    else:
        utility_fn = categorical_utility_regulariser

    # initialise model and guide
    model = BayesianRegression(1, 1)
    guide = AutoDiagonalNormal(model)
    print("Train covariate and target shape", train_X[:, 0].unsqueeze_(1).shape, train_Y.shape)
    model_and_guide_args = (train_X[:, 0].unsqueeze_(1), train_Y)
    # populate param_store
    guide(*model_and_guide_args)

    # define optimizer and loss function
    optimizer = torch.optim.Adam(guide.parameters(), **{"lr": lr, "betas": (0.90, 0.999)})
    loss_fn = pyro.infer.Trace_ELBO().differentiable_loss

    # compute loss
    elbo = []
    for i in range(num_iters + 1):

        loss = loss_fn(model, guide, *model_and_guide_args)

        if utility_calibration:
            if lazy_regularisation is None or i % lazy_regularisation == 0:
                loss = loss + utility_fn(model, guide, train_X, train_Y, target_X, target_Y,
                                                  unsupervised_utility,
                                                  EM=EM)
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
    if test_X is not None and test_Y is not None:
        test_args = (test_X, test_Y)
        with torch.no_grad():
            loss = loss_fn(model, guide, *test_args)
        print(f"Test ELBO is {loss/len(test_Y)}")
    return model, guide


def predictive_summary(samples):
    site_stats = {}
    for k, v in samples.items():
        site_stats[k] = {
            "mean": torch.mean(v, 0),
            "std": torch.std(v, 0),
            "5%": v.kthvalue(int(len(v) * 0.05), dim=0)[0],
            "95%": v.kthvalue(int(len(v) * 0.95), dim=0)[0],
        }
    return site_stats


def get_predictive(model, guide, X_data, return_sites=("linear.weight", "obs", "_RETURN")):
    with torch.no_grad():
        predictive = Predictive(model, guide=guide, num_samples=800,
                                return_sites=return_sites)
        samples = predictive(X_data)
        pred_summary = predictive_summary(samples)
    return pred_summary


def plot_predictive(pred_summary, params_for_plotting):
    data_generation.plot_data(show=False, **params_for_plotting)
    plt.suptitle("Posterior predictive distribution with 90% CI", fontsize=16)
    # plt.plot(non_african_nations["rugged"],
    #            non_african_nations["y_mean"])
    plt.fill_between(params_for_plotting["X"][:, 0],
                     pred_summary['obs']["5%"],
                     pred_summary['obs']["95%"],
                     alpha=0.5)
    # ax[0].plot(non_african_nations["rugged"],
    #            non_african_nations["true_gdp"],
    #            "o")
    # ax[0].set(xlabel="Terrain Ruggedness Index",
    #           ylabel="log GDP (2000)",
    #           title="Non African Nations")
    # idx = np.argsort(african_nations["rugged"])
    plt.show()


def main():
    run_train()


if __name__ == '__main__':
    main()
