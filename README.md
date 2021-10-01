# LCNMT
This repository is for my MSc. thesis project which will introduce loss calibrated variational inference for structured prediction problem, specifically to the NMT problem. I coined it Loss calibrated variational machine translation. 

| Method              | Job file to run to train         | Job file to run to evaluate | Requirements before training |
|---------------------|----------------------------------|-----------------------------|------------------------------|
| Baseline            | training_lcvnmt_baseline.job     | test_baseline.job           |                              |
| Baseline pretrained | training_baseline_pretrained.job | -                           |                              |
| BNN                 | training_bnn.job                 | test_bnn.job                |                              |
| BNN pretrained      | training_bnn_pretrained.job      | -                           |                              |
| Regulariser         | training_regulariser_final.job   | test_regulariser.job        | Baseline pretrained          |
| LCVNMT_{baseline}   | training_lcvnmt_baseline.job     | test_lcvnmt_baseline.job    | Baseline pretrained          |
| LCVNMT_{BNN}        | training_lcvnmt_bnn.job          | test_lcvnmt_bnn.job         | BNN pretrained               |
| LCVNMT_{BNN_{best}} | training_lcvnmt_bnn_best.job     | test_lcvnmt_bnn_best.job    | BNN                          |
| LCVNMT_{scratch}    | training_lcvnmt_scratch.job      | -                           |                              |
| LCVNMT_{sfe only}   | training_lcvnmt_sfe_only.job     | -                           | BNN pretrained               |
| LCVNMT_{big batch}  | training_lcvnmt_big_batch.job    | -                           | BNN pretrained               |
