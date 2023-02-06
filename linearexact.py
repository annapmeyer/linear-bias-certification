import interval
import perturbation
import dataset
import linear_eq
import exact_solver
import argparse
import numpy as np
import datetime


import warnings
warnings.filterwarnings("ignore", category=UserWarning)

''' Implements the exact certification procedure (section 4 of paper)'''

# perform matrix multiplication when theta (really theta^T) is intervals
# theta is 1 x d and x is d x 1
def eval_sample(theta, x):
    assert theta.shape[0] == 1
    assert x.shape[1] == 1
    assert theta.shape[1] == x.shape[0]

    total = interval.Interval(0, 0)
    for i in range(theta.shape[1]):
        total = interval.int_add(total, interval.int_mult(theta[0,i], interval.Interval(x[i,0],x[i,0])))
    return total

def main(args):
    dataset_name = args.dataset
    label_col = 'label'
    
    args.orig_target_val = args.target_val

    target = perturbation.Target(args.target_index, args.target_val, target_dir=args.target_dir)
    data_obj = dataset.Dataset(dataset_name, args.data_train, args.data_test, label_col, target, args)

    if ('demo' in dataset_name) or ('synth' in dataset_name):
        num_folds = 4
    elif ('mnist' in dataset_name) or ('fitz' in dataset_name):
        num_folds = 1
    else:
        num_folds = 10
    for i in range(num_folds):
        if num_folds == 1:
            x_train, y_train, x_test, y_test, scaler = data_obj.load_data(scale=args.scale)
        else:
            x_train, y_train, x_test, y_test, scaler = data_obj.load_data(fold=i, scale=args.scale)
        if scaler is not None and (args.target_index is not None):
            # create an array that is all 0's except for target_val in target_index
            ary = np.zeros(x_test.shape[1])
            ary[args.target_index] = args.orig_target_val
            args.target_val = float(scaler.transform(ary.reshape(1, -1))[0,args.target_index])

        target = perturbation.Target(args.target_index, args.target_val, target_dir=args.target_dir)
        lin_reg = linear_eq.Linear_Regression(dataset_name, x_train, y_train, x_test, y_test, args)
        solver = exact_solver.Exact_Solver(lin_reg, data_obj, target, args)

        results = []
        count = 0
        for j in range(args.num_to_run):
            if args.start + j >= len(x_test):
                continue
            valid = solver.find_perturbation(args.start + j)
            if not valid:
                continue
            count += 1
            results.append(solver.perturbation_size)

        print("{args:",args,"},{fold:",str(i),"},{count:",count,"},{results:",results,"}")
        break
   

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset')
    parser.add_argument('data_train') # complete filepath to train data
    parser.add_argument('data_test') # complete filepath to test data
    parser.add_argument('start', type=int)
    parser.add_argument('num_to_run', type=int)
    parser.add_argument('--label_per', default = 0, type = float) # for regression
    parser.add_argument('--robust_rad', default = 0, type = float)
    parser.add_argument('--target_index', default = None, type = int) # note, HMDA 10=race (1=black) and 11=gender (1=female)
    parser.add_argument('--target_val', default = None, type = float)
    # 1 if we can change -1 labels to 1, -1 for vice-versa
    # so 1 means we are assisting the 'disadvantaged' group and -1 is penalizing the 'advantaged' group
    parser.add_argument('--target_dir', type=int, default=1) 
    parser.add_argument('--demo_phi', default = None, type = int)
    parser.add_argument('--demo_phi_val', default = None)
    parser.add_argument('--neg_class', default = 0, type = int)
    parser.add_argument('--regularization', type = bool, default = 1)
    parser.add_argument('--max_label_flips', default = None, type=int)
    parser.add_argument('--regression', type = bool, default = 0)
    parser.add_argument('--tolerance', type = float, default = 0)
    parser.add_argument('--find_lambda', type = bool, default = 0)
    parser.add_argument('--scale', type=bool, default=False)
    args = parser.parse_args()

    if args.target_index == -1:
        args.target_index = None
        args.target_val = None
    if args.tolerance == -1:
        args.regularization = False
        args.tolerance = 0

    if (args.label_per != 0) or (args.robust_rad != 0):
        assert args.regression, "if you specify a label perturbation or robust radius, you must use regression"
    assert (args.label_per !=0) == (args.robust_rad != 0), "you must specify both label_per and robust_rad or neither"
    if args.target_index is not None:
        assert args.target_val is not None, "if you specify a target index you must specify a target value"

    main(args)
