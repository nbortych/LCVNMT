import torch
import numpy as np
import matplotlib.pyplot as plt


#
# def reverse_indices(source_indices, N):
#     total_indices = torch.linspace(0, N - 1, N).to(torch.int64)
#     total_indices, unique_idx = torch.cat((total_indices, source_indices)).unique(return_counts=True)
#     reversed_idices = total_indices[unique_idx == 1]
#     return reversed_idices


def assertion_error(num_categories, tuple_of_interest, argument_type="noise_variation"):
    assert argument_type in ['noise_variation', 'categories_proportions']
    objects_of_interest = {'noise_variation': 'noises', 'categories_proportions': 'categories proportions'}
    error = f"Sorry, you are trying to generate {num_categories} categories," \
            f"but have only specified {len(tuple_of_interest)} {objects_of_interest[argument_type]}." \
            f" Please, specify the remaining  {objects_of_interest[argument_type]} through {argument_type}" \
            f" argument by adding more floats to the tuple."
    return error


def renormalise_proportions(proportions):
    denominator = sum(proportions)
    categories_proportions = [proportion / denominator for proportion in proportions]
    return categories_proportions


def proportions_to_points(proportions, N):
    return list(map(lambda x: int(torch.clamp(torch.tensor(x), 0, 1) * N), proportions))


def generate_data(N=1000, noise_variation=(0.9, 0.3, 0.1), weight_std=1, bias_std=10, sine=True,
                  sine_covariate_shift=True, num_categories=3,
                  categories_proportions=(0.55, 0.25, 0.2)):
    """

    Args:
        N (int): Number of data points.
        noise_variation (tuple of floats): sigma for Gaussian noise added to points(of the first category).
        weight_std (float): sigma of Gaussian that samples the weight.
        bias_std (float): sigma of Gaussian that samples the bias.
        sine (bool): if True, sine function is the basis, otherwise linear.
        sine_covariate_shift (bool): if True, then all the categories except the first one will be after shift.
        num_categories (int): number of categories.
        categories_proportions (float or tuple): number between 0 and 1 that specifies proportion of points of the
                                                second category or a tuple with a convex hull with number of elements = num_categories.


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
    if num_categories > 1:
        # assertions to make sure that the parameters for categories are specified correctly
        assert type(noise_variation) == tuple, "Please, specify a tuple of noise variations"
        assert num_categories <= len(noise_variation), assertion_error(num_categories, noise_variation,
                                                                       argument_type="noise_variation")
        assert num_categories <= len(categories_proportions), assertion_error(num_categories, categories_proportions,
                                                                              argument_type="categories_proportions")
        assert sum(categories_proportions) == 1., f"Make sure that categories proportions add up to 1. Currently it's " \
                                                  f"{ sum(categories_proportions):.4f}."

        if num_categories < len(categories_proportions):
            print(f'There are less categories than proportions ({num_categories}<{len(categories_proportions)}).'
                  f' Renormalising first {num_categories} elements of the proportions. ')
            # renormalising
            categories_proportions = renormalise_proportions(categories_proportions[:num_categories])
        # just randomely distribute according to the proportion
        if not sine_covariate_shift:
            # get the number of points
            num_points_in_categories = proportions_to_points(categories_proportions, N)

            # make sure there are no rounding errors by adding the surplus to the first category
            if sum(num_points_in_categories) < N:
                num_points_in_categories[0] += N - sum(num_points_in_categories)
                # permute the indices for random split
            indices = torch.randperm(N)
            print(indices.shape)
        # distribute according to the position of the covariate
        else:
            shift_point = 5
            # get the training indices by splitting along the covariate
            train_idx, shift_idx = (X[:, 0] < shift_point).nonzero()[:, 0], (X[:, 0] > shift_point).nonzero()[:, 0]
            # the number of points in category 1 is determined by covariate, but the other categories don't have to
            shift_proportions = renormalise_proportions(categories_proportions[1:])
            num_points_in_categories = [train_idx.shape[0]] + proportions_to_points(shift_proportions,
                                                                                    shift_idx.shape[0])
            # shuffle shift indices
            shift_idx = shift_idx[torch.randperm(shift_idx.shape[0])]
            indices = torch.cat((train_idx, shift_idx), dim=0)

        # indices for different categories
        category_indices, total_points = [], 0
        for num_points in num_points_in_categories:
            category_indices.append(indices[total_points:total_points + num_points])
            total_points += num_points

        # encode categories as categorical data
        categories_tensor = torch.zeros(N).unsqueeze_(1)
        for i, category_idx in enumerate(category_indices):
            categories_tensor[category_idx] = i

        #  append the categories to the covariates
        # now second dimension encodes the category
        X = torch.cat((X, categories_tensor), dim=-1)

        # adding noise
        for category_idx, category_variation in zip(category_indices, noise_variation):
            Y[category_idx] += torch.distributions.Normal(0, category_variation).sample(Y[category_idx].shape)

    else:  # Append a single category
        X = torch.cat((X, torch.zeros(N).unsqueeze_(1)), dim=1)
        # just add noise to Y
        Y += torch.distributions.Normal(0, noise_variation[0]).sample(Y.shape)

    print(f"{'True function is sine' if sine else f'True function is linear and weight is: {w:.4}'}"
          f" and true bias is {b:.4}, true sigma is {noise_variation}")

    return_dict = {"predictors": X, "target": Y, "bias": b}
    if not sine:
        return_dict["weight"] = w
    return return_dict


def plot_data(X=None, Y=None, w=None, b=None, sine=True, show=True):
    assert X is not None and Y is not None, "Please pass the data."
    import matplotlib.colors as mcolors
    # get the categories (always last dimension)
    categories = X[:, -1].unique(sorted=True)
    indices = [[X[:, -1] == i, 0] for i in categories]
    # get the N colors from the color keymap, where N is the number of categories
    colors = list(mcolors.TABLEAU_COLORS.keys())[:len(categories)]

    # scatter the categories in different colors
    for idx, color in zip(indices, colors):
        plt.scatter(X[idx].numpy(), Y[idx[0]].numpy(), marker=".", color=color[4:])

    # plot the 'true' function on the whole range of the domain :)
    dummy_predictors = torch.linspace(X[:, 0].min(), X[:, 0].max(), steps=100)
    if sine:
        dummy_targets = np.sin(dummy_predictors) + b
    else:
        dummy_targets = w * dummy_predictors + b
    plt.plot(dummy_predictors.numpy(), dummy_targets.numpy(), color='r')

    if show:
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


def get_data(N=1000, sine=False, noise_variation=(0.4, 0.3, 0.1), plot=True,
             seed=None, covariate_shift=False, num_categories=2, categories_proportions=(0.5, 0.5)):
    if seed is not None:
        torch.manual_seed(seed)

    data = generate_data(N=N, sine=sine, noise_variation=noise_variation, sine_covariate_shift=covariate_shift,
                         num_categories=num_categories, categories_proportions=categories_proportions)
    X, Y, b = data["predictors"], data["target"], data["bias"]

    w = data["weight"] if not sine else None
    params_for_plotting = {"X": X, "Y": Y, "w": w, "b": b, "sine": sine}
    if plot:
        plot_data(**params_for_plotting)

    # returns data split into train, (target) and test sets
    if covariate_shift:  # todo fix
        train_X, train_Y, target_X, target_Y, test_X, test_Y = split_train_target_test_covariate_shift(X, Y)
    else:
        train_X, train_Y, test_X, test_Y = split_train_target_test_(X, Y, test_proportion=0.2)
        target_X, target_Y = None, None

    return train_X, train_Y, test_X, test_Y, params_for_plotting


def main():
    get_data(plot=True, sine=True)


if __name__ == "__main__":
    main()
