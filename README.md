# Certifying Data-Bias Robustness in Linear Regression

## Downloading data
The version of COMPAS we used is available in the data folder. The code to download Income and Income-Reg is in data/folktable.ipynb. The code to create the synthetic datasets is in data/generate synthetic data.ipynb. 

For MNIST 1/7, we used the standard data and used Rosenfeld et al.'s code to find a feature embedding (code available at http://www.cs.cmu.edu/~elan/). Their code trains a neural net on the data and saves the second-to-last layer as a feature embedding. You should save the embedder file as embedder.pth, the training data as xtrain_17.npy, and the test data as xtest_17.npy. 

## Running exact experiments
Run `python3 linearexact.py <dataset_name> <data_directory> start num_to_run`.

Additional command-line parameters are:
* regression - 1 if the dataset is a regression (not binary) dataset
* label_per - label perturbation ($$\Delta$$). Required if the regression flag is used.
* robust_rad - robustness radius ($$\epsilon$$). Required if the regression flag is used.
* tolerance - change the accuracy/robustness tradeoff. Tolerance value (and dataset) must be hard-coded in linear_eq.py
* demo_phi, demo_phi_val - if you want to check robustness for a particular subgroup, specify demo_phi=<index of feature>, demo_phi_val=<target value for feature>

### Interpreting the results

## Running approximate experiments
Run `python3 linear.py <dataset_name> <data_directory> --l=<num_label_flips>`

All the command-line parameters from linearexact.py are valid options, apart from `demo_phi` and `demo_phi_val`. Additionally, 
* l - number of label flips. Required unless there is a file `np_indices/<dataset_name>_indices.npy` containing a numpy array
* checkpoint - if true, load theta from a saved numpy file. Helpful to perform additional label-flips without starting from scratch. Will only work if the --l flag is larger than the previous number of label flips (i.e., the number of labels flipped for the stored theta). 
* maj_group_index, min_group_index, gender_index - feature index of the majority group (e.g., race), minority group, and gender
* maj_group_val, min_group_val, gender_val - feature value that indicates the training point is part of the majority group, minority group, or targeted gender group
* accuracy - flag to output the accuracy of a model. Will not run any robustness certification.
  
  
### Interpreting the results
