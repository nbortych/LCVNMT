import torch
import numpy as np
import matplotlib.pyplot as plt


#
# def reverse_indices(source_indices, N):
#     total_indices = torch.linspace(0, N - 1, N).to(torch.int64)
#     total_indices, unique_idx = torch.cat((total_indices, source_indices)).unique(return_counts=True)
#     reversed_idices = total_indices[unique_idx == 1]
#     return reversed_idices


def generate_data(N=1000, noise_variation=4, weight_std=1, bias_std=10, sine=True, categories=True,
                  category_2_proportion=0.2, category_2_noise_variation=8):
    """

    Args:
        N (int): Number of data points.
        noise_variation (float): sigma for Gaussian noise added to points(of the first category).
        weight_std (float): sigma of Gaussian that samples the weight.
        bias_std (float): sigma of Gaussian that samples the bias.
        sine (bool): if True, sine function is the basis, otherwise linear.
        categories (bool): if True, data is split into two categories with different noise.
        category_2_proportion (float): number between 0 and 1 that specifies proportion of points of the second category.
        category_2_noise_variation (float): sigma for Gaussian noise added to points of the second category.

    Returns:
        (dict): "predictors": the independent variable, covariates used to predict the Y.
                "target": the dependent variable.
                 "bias": true bias parameter.
                 if not sine:
                 "w": true weight parameter.
    """
    # generating bias
    b = torch.distributions.Normal(0, bias_std).sample()
    # generating data points

    # if sine function is the basis
    if sine:
        X = torch.linspace(2, 2.2 * 3.14, N)
        Y = torch.sin(X) + b
    # if linear function is the basis
    else:
        w = torch.distributions.Normal(0, weight_std).sample()
        X = torch.randn(N)
        Y = w * X + b
    X = X.unsqueeze_(1)

    # if introducing a categorical variable
    if categories:
        # get the number of points
        num_points_in_category_2 = int(torch.clamp(torch.tensor(category_2_proportion), 0, 1) * N)
        category_threshold = N - num_points_in_category_2
        # permute the indices
        indices = torch.randperm(N)
        # first element are the first category, second is the second category
        category_indices = [indices[:category_threshold], indices[category_threshold:]]

        # append the categories to the covariates
        categories_tensor = torch.ones(N).unsqueeze_(1)
        categories_tensor[category_indices[0]] = 0
        # now second dimension is 0 if category 1 and 1 if category 2
        X = torch.cat((X, categories_tensor), dim=1)

        # adding noise
        for category_idx, category_variation in zip(category_indices,
                                                    [noise_variation, category_2_noise_variation]):
            Y[category_idx] += torch.distributions.Normal(0, category_variation).sample(Y[category_idx].shape)

    else:  # just add noise to Y
        Y += torch.distributions.Normal(0, noise_variation).sample(Y.shape)
    print(
        f"{'True function is sine' if sine else f'True function is linear and weight is: {w:.4}'} and true bias is {b:.4}, true sigma is {noise_variation}")

    return_dict = {"predictors": X, "target": Y, "bias": b}
    if not sine:
        return_dict["weight"] = w
    return return_dict


def plot_data(X, Y, w=None, b=None, sine=True, categories=True):
    # scatter the two categories in different colors
    if categories:
        indices = [[X[:, 1] == 0, 0], [X[:, 1] == 1, 0]]
        colors = ["blue", "red"]
        for idx, color in zip(indices, colors):
            plt.scatter(X[idx].numpy(), Y[idx[0]].numpy(), marker=".", color=color)
    # just scatter them
    else:
        plt.scatter(X.numpy(), Y.numpy(), marker=".")

    # plot the 'true' function on the whole range of the domain :)
    dummy_predictors = torch.linspace(X[:, 0].min(), X[:, 0].max(), steps=4)
    if sine:
        dummy_targets = np.sin(dummy_predictors) + b
    else:
        dummy_targets = w * dummy_predictors + b
    plt.plot(dummy_predictors.numpy(), dummy_targets.numpy(), color='r')

    plt.show()


def split_train_target_test_covariate_shift(X, Y, shift_point=5, test_proportion=0.5):
    """
    Splits the dataset into train, test (and target) data.
    Args:
        X (tensor): covariate/independent data in shape (N,num_features)
        Y (tensor): dependent data in shape of (N)
        shift_point (float): point on the covariate that determines where datashift happens
        test_proportion (float): proportion of the shifted points that go to testing.
    Returns:

    """
    # get the training indices by splitting along the covariate
    train_idx = (X[:, 0] < shift_point).nonzero()[:, 0]
    # get the target and test indices
    idx = (X[:, 0] > shift_point).nonzero()[:, 0]

    # half of the test set will be unseen
    test_size = int(idx.shape[0] * test_proportion)
    # randomly shuffle the indices
    target_and_test_indices = torch.randperm(idx.shape[0])
    # assign test_size points to the test set and the rest to target set
    test_idx, target_idx = idx[target_and_test_indices[:test_size]], idx[target_and_test_indices[test_size:]]
    # get the points
    target_X, target_Y = X[target_idx], Y[target_idx],
    test_X, test_Y = X[test_idx], Y[test_idx]
    train_X, train_Y = X[train_idx], Y[train_idx]

    tr_n, ta_n, te_n = train_Y.shape[0], target_Y.shape[0], test_Y.shape[0]
    print(f"Got a dataset with {tr_n} training points, {ta_n} target points"
          f" and {te_n} testing points, for a total of {sum([tr_n, ta_n, te_n])} points")
    return train_X, train_Y, target_X, target_Y, test_X, test_Y


def split_train_target_test_(X, Y, test_proportion=0.2):
    """
    Splits the data into train and test set.
    Args:
        X (tensor): covariate/independent data in shape (N,num_features)
        Y (tensor): dependent data in shape of (N)
        test_proportion (float): proportion of the points that go to testing.

    Returns:

    """
    N = X.shape[0]
    indices = torch.randperm(N)
    test_num_points = int(N * test_proportion)
    test_X, test_Y = X[indices[:test_num_points]], Y[indices[:test_num_points]]
    train_X, train_Y = X[indices[test_num_points:]], Y[indices[test_num_points:]]

    tr_n, te_n = train_Y.shape[0], test_Y.shape[0]
    print(f"Got a dataset with {tr_n} training points and {te_n} testing points, "
          f"for a total of {tr_n+te_n} points")
    return train_X, train_Y, test_X, test_Y


def get_data(N=1000, sine=False, categories=False, noise_variation=0.4, noise_variation_2=2, plot=True, seed=None):
    if seed is not None:
        torch.manual_seed(seed)

    data = generate_data(N=N, noise_variation=noise_variation, sine=sine, categories=categories,
                         category_2_noise_variation=noise_variation_2)
    X, Y, b = data["predictors"], data["target"], data["bias"]

    w = data["weight"] if not sine else None

    if plot:
        plot_data(X, Y, w, b, sine=sine, categories=categories)

    # returns data split into train, (target) and test sets
    if sine:
        train_X, train_Y, target_X, target_Y, test_X, test_Y = split_train_target_test_covariate_shift(X, Y)
        return train_X, train_Y, target_X, target_Y, test_X, test_Y
    else:
        train_X, train_Y, test_X, test_Y = split_train_target_test_(X, Y, test_proportion=0.2)
        return train_X, train_Y, None, None, test_X, test_Y


def main():
    get_data(plot=True, categories=True, sine=True)


if __name__ == "__main__":
    main()
