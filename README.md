# Certifying Data-Bias Robustness in Linear Regression

## Downloading data
The version of COMPAS we used is available in the data folder. The code to download Income and Income-Reg is in `data/folktable.ipynb`. The code to create the synthetic datasets is in `data/generate synthetic data.ipynb`. 

For MNIST 1/7, we used the standard data and used Rosenfeld et al.'s code to find a feature embedding (code available at http://www.cs.cmu.edu/~elan/). Their code trains a neural net on the data and saves the second-to-last layer as a feature embedding. You should save the embedder file as `embedder.pth`, the training data as `xtrain_17.npy`, and the test data as `xtest_17.npy`. 

## Running exact experiments
Run `python3 linearexact.py <dataset_name> <data_directory> start num_to_run`.

Additional command-line parameters are:
* `regression` - 1 if the dataset is a regression (not binary) dataset
* `label_per` - label perturbation ($$\Delta$$). Required if the regression flag is used.
* `robust_rad` - robustness radius ($$\epsilon$$). Required if the regression flag is used.
* `tolerance` - change the accuracy/robustness tradeoff. Tolerance value (and dataset) must be hard-coded in `linear_eq.py`.
* `demo_phi`, `demo_phi_val` - if you want to check robustness for a particular subgroup, specify `demo_phi=<index of feature>`, `demo_phi_val=<target value for feature>`

#### Example prompts
`python3 linearexact.py income data 0 20 --tolerance=2` find the minimum number of label perturbations necessary for non-robustness for the first 20 points in the Income data. Use $$ \lambda$$ so that the accuracy is within 2\% of optimal.

`python3 linearexact.py compas data 10 10 --demo_phi=0 --demo_phi_val=1` find the minimum number of label-perturbations necessary for non-robustness for indices 10-19 of the COMPAS dataset. Limit the label bias so that only labels corresponding to White people (feature 0, value 1) may be perturbed.

`python3 linearexact.py income_reg data 0 20 --label_per=4000 --epsilon=2000` find the minimum number of label perturbations of size 4000 necessary to change the predictions by more than 2000, for the first 20 test points in the Income-Reg dataset.

### Interpreting the results
For each fold/test point pair, the code will print something like: 

` {args: Namespace(datadir='chtc', dataset='income', demo_phi=None, demo_phi_val=None, find_lambda=0, label_per=0, max_label_flips=None, neg_class=0, num_to_run=5, regression=0, regularization=0, robust_rad=0, start=0, target_index=None, target_val=None, tolerance=None) },{fold: 0 },{count: 5 },{results: [1024, 59, 216, 718, 187] } `

The first tuple just summarizes the arguments and can be helpful if analyzing a large set of results. The final tuple, e.g., `{results: [1024, 59, 216, 718, 187]}`, summarizes the minimum number of label perturbations necessary to change the predictions. That is, test point 0 needed 1024 label perturbations, test point 1 required 59, and so on. If any result values are -1, this means that there is no label perturbation that will change the prediction. 

## Running approximate experiments
Run `python3 linear.py <dataset_name> <data_directory> --l=<num_label_flips>`

All the command-line parameters from linearexact.py are valid options, apart from `demo_phi` and `demo_phi_val`. Additionally, 
* `l` - number of label flips. Required unless there is a file `np_indices/<dataset_name>_indices.npy` containing a numpy array
* `checkpoint` - if true, load theta from a saved numpy file. Helpful to perform additional label-flips without starting from scratch. Will only work if the --l flag is larger than the previous number of label flips (i.e., the number of labels flipped for the stored theta). 
* `maj_group_index`, `min_group_index`, `gender_index` - feature index of the majority group (e.g., race), minority group, and gender
* `maj_group_val`, `min_group_val`, `gender_val` - feature value that indicates the training point is part of the majority group, minority group, or targeted gender group
* `accuracy` - flag to output the accuracy of a model. Will not run any robustness certification.
  
  
### Interpreting the results
For each fold/label perturbation amount pair, the code will output something like:

` {args: Namespace(accuracy=False, checkpoint=0, datadir='chtc', dataset='compas', find_lambda=False, gender_index=7, gender_val='1', l=20, label_per=0, maj_group_index=0, maj_group_val='1', min_group_index=1, min_group_val='1', neg_class=0, regression=0, regularization=1, robust_rad=0, target_index=None, target_val=None, tolerance=0) }, {fold: 0 }, {labels: 20 }, {robust: 0.616504854368932 }, {robust_pct_men:  0.6833333333333333 }, {robust_pct_women: 0.6004016064257028 },{robust_pct_maj: 0.634703196347032 }, {robust_pct_min: 0.596875 } `

The relevant pieces are `robust: ` and `robust_pct_<demo group>` where the different demographic groups correspond to the features and values specified in the command line parameters. The robustness rates are over the entire dataset, i.e., they are the fraction of samples that are certifiably-robust for the given number of label perturbations. 


