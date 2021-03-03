from sklearn.model_selection import ParameterGrid
import wandb

import LCDA_BLR


def experiment_local_fast():
    """
    Runs the experiment according to params specified in the param_grid.
    """
    param_grid = {'calibration': [True, False], 'oracle': [True, False], 'weights_per_category': [{0: 0, 1: 1}]}
    for param in ParameterGrid(param_grid):
        if param['calibration'] == False and param['oracle'] == param_grid['oracle'][0]:
            continue
        print(f"THE PARAMETERS ARE \n {param} \n {'___'*60}")
        LCDA_BLR.run_train(utility_calibration=param['calibration'], weights_per_category=param['weights_per_category'],
                           oracle=param['oracle'])


def wandb_experiment():
    """
    Makes the wandb sweep according to params in sweep_config.
    """
    sweep_config = {
        'method': 'grid',
        'parameters': {
            'utility_calibration': {
                'values': [True, False]
            },
            'weights_per_category': {
                'values': [{"0": 0, "1": 1}, {"0": 1, "1": 1}]
            },
            'sine': {
                'values': [True, False]
            },
            'oracle': {
                'values': [True, False]
            }
        }
    }

    sweep_id = wandb.sweep(sweep_config, project="lcda_blr")
    wandb.agent(sweep_id, function=wandb_run)


def wandb_run(config=None):
    """
    One run of the experiment. Gets config, runs the experiment and logs all the results
    Args:
        config (dict): parameters.

    Returns:

    """
    # init the project
    wandb.init(project="lcda_blr", config=config)
    config = wandb.config
    print(config)
    # set up parameters
    if config['sine']:
        polynomial_degree = 3
        num_iters = 4000
    else:
        polynomial_degree = 1
        num_iters = 1000
    # skip meaningless combinations
    if (not config['utility_calibration'] and config['weights_per_category'] == {"0": 0, "1": 1}) or\
        (not config['utility_calibration'] and config['oracle']):
        return

    # run experiment
    results = LCDA_BLR.run_train(sine=config['sine'], utility_calibration=config['utility_calibration'],
                                 weights_per_category=config['weights_per_category'],
                                 oracle=config['oracle'], num_iters=num_iters,  # config['oracle']
                                 loss_type="SE", polynomial_degree=polynomial_degree, guide_type="diag",
                                 show_plot=False, test=0)
    # log the elbo by iteration
    for elbo in results['elbo']:
        wandb.log({"elbo": elbo})
    del results['elbo']

    # log the variational parameters
    for param_name, param_values in results['variational_params'].items():
        for i, param_value in enumerate(param_values):
            wandb.log({f'{param_name}_{i}': param_value})
    del results['variational_params']

    # turn the plot into an image
    results['predictive_plt'] = wandb.Image(results['predictive_plt'])

    # log everything
    wandb.log(results)
    # wandb.log({"Test Accuracy": correct / total, "Test Loss": loss})


def main():
    print("AAAA")
    wandb_experiment()
    # experiment_local_fast()

    # LCDA_BLR.run_train(utility_calibration=True, weights_per_category={0: 0, 1: 10})


#


if __name__ == '__main__':
    main()

#  NO CALIBRATION
# SE LOSS
# Final parameters are [('AutoDiagonalNormal.loc' tensor([-2.5004, -0.1268,  3.6603], requires_grad=True)),
# ('AutoDiagonalNormal.scale', tensor([0.0332, 0.0099, 0.0293], grad_fn=<AddBackward0>))]
# Final  ELBO is 1.17
# Total train L2 loss is 4271.47 and on category 2 train L2 loss is 1261.92
# Test ELBO is 0.5699791312217712
# Total test L2 loss is 1097.7515869140625 and on category 2 test L2 loss is 264.1929931640625
# AE LOSS
# Total train AE loss is 5030.97 and on category 2 train AE loss is 1516.77
# Total test AE loss is 1277.746 and on category 2 test AE loss is 341.802


# FULL CALIBRATION
# SE Loss
# Final parameters are [('AutoDiagonalNormal.loc', tensor([-2.7404, -0.1283,  3.6845], requires_grad=True)),
#  ('AutoDiagonalNormal.scale', tensor([0.0546, 0.0114, 0.0178], grad_fn=<AddBackward0>))]
# Final Utility calibrated ELBO is 4.61
# Final ELBO is 1.21
# Total train L2 loss is 2659.26 and on category 2 train L2 loss is 788.30
# Test ELBO is 0.6652511358261108
# Total test L2 loss is 691.6593627929688 and on category 2 test L2 loss is 210.7091522216797
# AE loss
# Final parameters are [('AutoDiagonalNormal.loc',  tensor([-3.2159, -0.0305,  3.2533], requires_grad=True)),
#  ('AutoDiagonalNormal.scale', tensor([0.0397, 0.0038, 0.0106], grad_fn=<AddBackward0>))]
# Final Utility calibrated ELBO is 4.99
# Final ELBO is 1.92
# Total train AE loss is 2484.68 and on category 2 train AE loss is 730.91
# Test ELBO is 1.259
# Total test AE loss is 625.120 and on category 2 test AE loss is 183.765

# CATEGORY 2 CAlIBRATION
# SE loss
# Final parameters are [('AutoDiagonalNormal.loc',  tensor([-2.7292, -0.1284,  3.6841], requires_grad=True)),
#  ('AutoDiagonalNormal.scale', tensor([0.0533, 0.0108, 0.0197], grad_fn=<AddBackward0>))]
# Final Utility calibrated ELBO is 2.23
# Final ELBO is 1.20
# Total train L2 loss is 2710.39 and on category 2 train L2 loss is 804.19
# Test ELBO is 0.6589685678482056
# Total test L2 loss is 706.0341186523438 and on category 2 test L2 loss is 214.82980346679688
# AE loss
# Final parameters are [('AutoDiagonalNormal.loc',  tensor([-2.9141, -0.0525,  3.3654], requires_grad=True)),
#  ('AutoDiagonalNormal.scale', tensor([0.0328, 0.0046, 0.0175], grad_fn=<AddBackward0>))]
# Final Utility calibrated ELBO is 2.62
# Final ELBO is 1.36
# Total train AE loss is 3331.48 and on category 2 train AE loss is 979.72
# Test ELBO is 0.823
# Total test AE loss is 837.731 and on category 2 test AE loss is 246.157


# CATEGORY 2 X10 CAlIBRATION
# Final ELBO is 11.30
# Final train L2 loss is 2499.5908203125 and on category 2 train L2 loss is 778.4462280273438
# Test ELBO is 0.6490957736968994
# Test L2 loss is 680.1661376953125 and on category 2 test L2 loss is210.0272979736328
