import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import pyro
import torch
import numpy as np
import matplotlib.pyplot as plt

from pyro.infer.autoguide import AutoMultivariateNormal, AutoDiagonalNormal
from pyro.infer import Predictive
import data_generation
from BLR import BayesianRegression


# todo
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


# for polynomial regression
def designmatrix(x, M):
    """
    Generates design matrix for polynomial regression. A matrix with each column being an incremental power of x up to M.
    Args:
        x (tensor): one-dimensional covariates.
        M (int): the order of polynomial.

    Returns:
        phi (tensor): a N by M matrix, where each column is x to the power of column_number.
    """
    phi = torch.empty(M, x.shape[0])
    for i in range(M):
        phi[i] = torch.pow(x, i + 1).reshape(x.shape[0])

    return phi.T


def transform_loss_to_utility(loss, method="exp", gamma=1):
    """
    Transforms a loss into a utility. Needed not to break the log and JI.
    Args:
        loss (tensor): loss value to be transformed.
        method (str): type of transformation. So far only exponential is used.
        gamma (float): by how much to scale the loss.

    Returns:
        utility (tensor): utility version of the loss
    """
    return torch.exp(-loss * gamma)


#
# def utility_regulariser_classical_lcvi(model, guide, train_X, train_Y, x_target, y_target=None,
#                                        unsupervised_utility=True, S=2, M=5,
#                                        loss_type="SE", EM=False):
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
#     args = data[:, 0].unsqueeze(1)
#     N = args.shape[0]
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


def utility_regulariser_categorical(model, guide, train_X, train_Y,
                                    number_paramater_samples=5, number_predictive_samples=2, weights_tensor=None,
                                    oracle=False, loss_type="AE", EM=False, loss_to_utility=False):
    """
    Computes the
        1. Irrespective of y: \int q_{\lambda}(\theta) \log \int p(y \mid \theta, \mathcal{D}) u(y, h) d y d \theta
                1. Sample S \theta from q(\theta).
                2. Sample N predictive ys based on p(y|x,\theta).
                3. Estimate h: closed form for now
                4. Compute u(y,h)
    Args:
        model: pyro model specifying prior and likelihood.
        guide: pyro guide that allows to sample from variational parameters.
        train_X (tensor): specifies the covariates. Shape N,D, where d is polynomial degree.
        train_Y (tensor): specifies the targets.
        number_paramater_samples (int): number of times variational parameters are sampled.
        number_predictive_samples (int): number of times the predictive distribution is sampled with 1 param sample.
        weights_tensor (tensor or int): specifies per data point weight to weight the categories.
        oracle (bool): if True, the actual data will be used as the decision, otherwise Bayes optimal decision.
        loss_type (str): one of "AE" : L1 loss, "SE": L2 loss.
        EM (bool): not implemented. if true, h will be computed by using different y.
        loss_to_utility (bool): if true, loss will be converted to utility

    Returns:
        loss (tensor):  utility loss under guide
    """

    assert loss_type in ['SE', 'AE'], "Please, specify a valid loss."

    if type(weights_tensor) == int:
        pass
    elif type(weights_tensor) == type(torch.tensor(1)):
        pass
    else:
        weights_tensor = 1

    # check for dimensionality of data
    assert len(
        train_X.shape) == 2, f"Make sure the train_X is of the correct shape: (N,d). Currently it's shaped {train_X.shape}"

    args = train_X

    N = args.shape[0]
    loss = 0.0
    predictive_samples = torch.empty(number_paramater_samples, number_predictive_samples, N)
    predictive = Predictive(model, posterior_samples=None, guide=guide,
                            num_samples=number_predictive_samples, return_sites=['obs'])
    # Compute predictive samples
    for s in range(number_paramater_samples):
        predictive_samples[s] = predictive(args)['obs']
        # print(predictive_samples[s].shape)
    total_number_of_samples = number_paramater_samples * number_predictive_samples
    samples_n_shape = (total_number_of_samples, N)
    # Compute decision.
    if loss_type == "SE" and not oracle:
        # The bayes estimator for this loss function is the mean of predictive posterior samples.
        # In code of tkusmierczyk in LCVI directory, he uses mean with respect to both theta and predictive
        with torch.no_grad():
            h = torch.mean(predictive_samples.view(samples_n_shape), dim=0)
    elif loss_type == "AE" and not oracle:
        with torch.no_grad():
            h = torch.median(predictive_samples.view(samples_n_shape), dim=0)[0]
    elif oracle:
        h = train_Y

    # Compute loss
    if loss_type == "SE":
        loss = torch.sum(weights_tensor *
                         (predictive_samples.view(samples_n_shape) - h) ** 2) / total_number_of_samples
    elif loss_type == "AE":
        loss = torch.sum(weights_tensor *
                         torch.abs(predictive_samples.view(samples_n_shape) - h)) / total_number_of_samples
    # if we want to transform it to utility
    if loss_to_utility:
        loss = - transform_loss_to_utility(loss)
    return loss


def get_utility_weights(data, weights_per_category={'0': 0, "1": 1}):
    """
    Function that generated weight tensor to weight importance of categories. If all weights are the same, returns an int,
    otherwise, returns per data point weight.
    Args:
        data (tensor): covariates with the categories.
        weights_per_category (dict): key specifies the category and value specifies the weight.

    Returns:
        weight (int or tensor)
    """
    # number of categories in data and number of specified weights
    num_categories = len(data[:, 1].unique())
    num_weights = len(weights_per_category.keys())
    # compute the weigths tensor for each category
    assert num_categories <= num_weights, f"Please add {num_categories-num_weights} categories to the" \
                                          f" weights_per_category dictionary"

    # check if all the weights are the same. If yes, then pass just that weight
    all_weights = set(list(weights_per_category.values())[:num_categories])
    weight = next(iter(all_weights)) if len(all_weights) == 1 else None

    if weight is None:
        # specify the weights
        utility_weights = torch.zeros_like(data[:, 0])
        for category in range(num_categories):
            utility_weights[data[:, 1] == category] = weights_per_category[str(category)]
    else:
        utility_weights = weight
    return utility_weights


def run_train(sine=True, covariate_shift=False, num_categories=2, num_iters=5000,
              seed=42, test=1, weights_per_category={'0': 1, '1': 1}, utility_calibration=1, loss_type="SE",
              loss_to_utility=False, polynomial_degree=1, oracle=1, guide_type="diag", show_plot=True):
    """

    Args:
        sine (bool): if True, sine function is the basis, otherwise linear.
        covariate_shift (bool): if True, then all the categories except the first one will be after shift.
        num_categories (int): number of categories.
        num_iters (int): number of iterations the training will run for.
        seed (int): seed for the RNG.
        test (bool): if True, only 10 iterations will be used.
        weights_per_category (dict): key specifies the category and value specifies the weight.
        utility_calibration (bool): if True, utility regulariser will be used.
        loss_type (str): one of "AE" : L1 loss, "SE": L2 loss.
        loss_to_utility (bool): if true, loss will be converted to utility
        polynomial_degree (int): the maximal power to which the data will be raised for polynomial regression
        oracle (bool): if True, the actual data will be used as the decision, otherwise Bayes optimal decision.
        guide_type (str):  one of ['diag', 'multi'] which autoguide to use.
        show_plot (bool): if True, the predictive posterior around the data will be plotted.

    Returns:
        results (dict): results of the optimisation
    """
    train_X, train_Y, test_X, test_Y, params_for_plotting = data_generation.get_data(N=1000,
                                                                                     sine=sine,
                                                                                     noise_variation=(0.2, 0.9),
                                                                                     categories_proportions=(
                                                                                         0.7, 0.3),
                                                                                     plot=False,
                                                                                     seed=seed,
                                                                                     covariate_shift=covariate_shift,
                                                                                     num_categories=num_categories)
    print(f"Weights are {weights_per_category}")

    utility_weights_tensor = get_utility_weights(train_X, weights_per_category=weights_per_category)

    model, guide, results = train(train_X, train_Y, num_iters, test_X, test_Y,
                                  seed=seed, test=test, weights_tensor=utility_weights_tensor,
                                  utility_calibration=utility_calibration,
                                  loss_type=loss_type, loss_to_utility=loss_to_utility,
                                  polynomial_degree=polynomial_degree, oracle=oracle, guide_type=guide_type)

    covariates = designmatrix(params_for_plotting["X"][:, 0], polynomial_degree) if polynomial_degree != 1 else \
        params_for_plotting["X"][:, 0].unsqueeze_(1)

    pred_summary = get_predictive(model, guide, covariates, return_sites=["obs"])
    result_plot = plot_predictive(pred_summary, params_for_plotting, plot=show_plot)
    results['predictive_plt'] = result_plot

    return results


def train(train_X, train_Y, num_iters=1000, test_X=None, test_Y=None, lr=0.01, seed=42,
          test=0, utility_calibration=True, EM=False, polynomial_degree=2, oracle=False,
          lazy_regularisation=None, weights_tensor=None, loss_type="SE", loss_to_utility=False, guide_type="diag"):
    """

    Args:
        train_X (tensor): covariate. first dimension is the data, second are the categories.
        train_Y (tensor): targets.
        num_iters (int): number of iterations the training will run for.
        test_X (tensor): same as train, but for testing
        test_Y (tensor): test targets.
        lr (float): learning rate for Adam.
        seed (int): seed for the RNG.
        test (bool): if True, only 10 iterations will be used.
        utility_calibration (bool): if True, utility regulariser will be used.
        EM (bool): not implemented. if true, h will be computed by using different y.
        polynomial_degree (int): the maximal power to which the data will be raised for polynomial regression
        oracle (bool): if True, the actual data will be used as the decision, otherwise Bayes optimal decision.
        lazy_regularisation (int): if passed, will use regularisation every lazy_regularisation iterations.
        weights_tensor (int or tensor): how to weight the datapoints in utility regularisation
        loss_type (str): one of "AE" : L1 loss, "SE": L2 loss.
        loss_to_utility (bool): if true, loss will be converted to utility
        guide_type (str):  one of ['diag', 'multi'] which autoguide to use.

    Returns:

    """
    # PREPARATIONS for training
    # clear param store and seed
    pyro.clear_param_store()
    if seed is not None:
        torch.manual_seed(seed)
    if test:
        num_iters = 10

    # initialise model and guide
    model = BayesianRegression(polynomial_degree, 1)
    # guides
    if guide_type == "diag":
        guide = AutoDiagonalNormal(model)
    else:
        guide = AutoMultivariateNormal(model)
    # process inputs
    print("Train covariate and target shape", train_X[:, 0].unsqueeze_(1).shape, train_Y.shape)
    model_and_guide_args = [designmatrix(train_X[:, 0].unsqueeze_(1), polynomial_degree), train_Y]
    # populate param_store
    guide(*model_and_guide_args)

    # define optimizer and loss function
    optimizer = torch.optim.Adam(guide.parameters(), **{"lr": lr, "betas": (0.90, 0.999)})
    loss_fn = pyro.infer.Trace_ELBO().differentiable_loss

    # type of calibration
    if utility_calibration:
        type_of_calibration = "Utility calibrated" if loss_to_utility else "Loss calibrated"
    else:
        type_of_calibration = ""

    # compute loss
    elbo = []
    for i in range(num_iters + 1):

        loss = loss_fn(model, guide, *model_and_guide_args)

        if utility_calibration:
            if lazy_regularisation is None or i % lazy_regularisation == 0:
                loss = loss + utility_regulariser_categorical(model, guide, model_and_guide_args[0], train_Y,
                                                              EM=EM, weights_tensor=weights_tensor, loss_type=loss_type,
                                                              loss_to_utility=loss_to_utility, oracle=oracle)
        loss.backward()
        # take a step and zero the parameter gradients
        optimizer.step()
        optimizer.zero_grad()
        iter_loss = loss.item() / train_X.shape[0]
        if i % (1 if test else 100) == 0:
            elbo.append(iter_loss)
            print(f"[{i :4d}/{num_iters}] {type_of_calibration} ELBO: {iter_loss:.4f}")

    # RESULTS PRINTING + LOGGING
    results = {}
    results['elbo'] = elbo
    print(f"Final parameters are {list(pyro.get_param_store().items())}")
    print(f"Final {type_of_calibration} ELBO is {elbo[-1]:.2f}")
    # logging
    results['final_elbo'] = elbo[-1]

    variational_params = {param_name: param_value.detach().numpy() for param_name, param_value in
                          pyro.get_param_store().items()}
    results['variational_params'] = variational_params

    # if utility_calibration:
    with torch.no_grad():
        final_elbo = loss_fn(model, guide, *model_and_guide_args) / train_X.shape[0]
    print(f"Final {type_of_calibration} ELBO is {final_elbo:.2f}")
    results['utility_elbo'] = final_elbo.item()

    # Evaluating only the (loss/)utility
    weights_tensor_second_category = get_utility_weights(train_X, weights_per_category={'0': 0, '1': 1})

    for weights_t, result_name in zip([1, weights_tensor_second_category],
                                      ['final_train_loss_all_categories', 'final_train_loss_second_category']):
        results[result_name] = utility_regulariser_categorical(model, guide, model_and_guide_args[0], train_Y,
                                                               weights_tensor=weights_t, loss_type=loss_type,
                                                               oracle=True).item()
    print(
        f"Total train {loss_type} {type_of_calibration}  is "
        f"{results['final_train_loss_all_categories']:.2f} "
        f"and on category 2 train {loss_type} {type_of_calibration}  is "
        f"{results['final_train_loss_second_category']:.2f}")

    # todo do holdout log likelihood on , test_X, test_Y
    if test_X is not None and test_Y is not None:
        # test_args = (test_X[:, 0].unsqueeze_(1), test_Y)
        # mask_category_1 = test_X[:, 1] == 1
        # test_args = (test_X[mask_category_1, 0].unsqueeze_(1), test_Y[mask_category_1])
        test_args = [designmatrix(test_X[:, 0], polynomial_degree), test_Y]

        with torch.no_grad():
            loss = loss_fn(model, guide, *test_args)
        weights_tensor_second_category = get_utility_weights(test_X, weights_per_category={'0': 0, '1': 1})

        for weights_t, result_name in zip([1, weights_tensor_second_category],
                                          ['final_test_loss_all_categories', 'final_test_loss_second_category']):
            results[result_name] = utility_regulariser_categorical(model, guide, test_args[0], test_Y,
                                                                   weights_tensor=weights_t, loss_type=loss_type,
                                                                   oracle=True).item()
        print(f"Test ELBO is {loss/len(test_Y):.3f}")
        print(
            f"Total test {loss_type} {type_of_calibration} is "
            f"{results['final_test_loss_all_categories']:.3f}"
            f" and on category 2 test {loss_type} {type_of_calibration}  is "
            f"{results['final_test_loss_second_category']:.3f}")

    return model, guide, results


def predictive_summary(samples):
    """
    Summarises predictive samples.
    Args:
        samples (dict): where key is the site and value is tensor with samples.

    Returns:

    """
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
    """
    Gets the predictive samples from guide, conditioned on X_data for the values of the return_sites.
    Args:
        model: pyro model specifying prior and likelihood.
        guide: pyro guide that allows to sample from variational parameters.
        X_data (tensor): covariates
        return_sites (tuple): which sites to return

    Returns:

    """
    with torch.no_grad():
        predictive = Predictive(model, guide=guide, num_samples=800,
                                return_sites=return_sites)
        samples = predictive(X_data)
        pred_summary = predictive_summary(samples)
    return pred_summary


def plot_predictive(pred_summary, params_for_plotting, plot=True):
    """
    Plots the 90% CI predictive posterior distribution and the original data.
    Args:
        pred_summary (dict): summary of the predictive posterior.
        params_for_plotting (list): parameters to plot the data
        plot (bool): if True, will be shown

    Returns:

    """
    plt.clf()
    data_generation.plot_data(show=False, **params_for_plotting)
    plt.suptitle("Posterior predictive distribution with 90% CI", fontsize=16)
    predictive_tensors = torch.cat((params_for_plotting["X"][:, 0].unsqueeze(1), pred_summary['obs']['5%'].unsqueeze(1),
                                    pred_summary['obs']['95%'].unsqueeze(1), pred_summary['obs']['mean'].unsqueeze(1)),
                                   dim=1)
    predictive_tensors = predictive_tensors[predictive_tensors[:, 0].sort()[1]]

    plt.fill_between(predictive_tensors[:, 0],
                     predictive_tensors[:, 1],
                     predictive_tensors[:, 2],
                     alpha=0.5, color='tab:blue')

    plt.plot(predictive_tensors[:, 0], predictive_tensors[:, 3], color="g")

    if plot:
        plt.show()

    result_plot = plt
    return result_plot


def main():
    run_train()


if __name__ == '__main__':
    main()
