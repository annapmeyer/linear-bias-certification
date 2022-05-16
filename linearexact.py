import interval
import perturbation
import dataset
import linear_eq
import exact_solver
import argparse
import numpy as np

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

    train_filename = args.datadir + "/train_" + dataset_name + "_"
    test_filename =  args.datadir + "/train_" + dataset_name + "_"

    target = perturbation.Target(args.target_index, args.target_val)
    data_obj = dataset.Dataset(dataset_name, train_filename, test_filename, label_col, target, args)

    if ('demo' in dataset_name) or ('synth' in dataset_name):
        num_folds = 4
    elif 'mnist' in dataset_name:
        num_folds = 1
    else:
        num_folds = 10
    for i in range(num_folds):
        x_train, y_train, x_test, y_test = data_obj.load_data(i)
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
   

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset')
    parser.add_argument('datadir')
    parser.add_argument('start', type=int)
    parser.add_argument('num_to_run', type=int)
    parser.add_argument('--label_per', default = 0, type = float)
    parser.add_argument('--robust_rad', default = 0, type = float)
    parser.add_argument('--target_index', default = None, type = int)
    parser.add_argument('--target_val', default = None, type = float)
    parser.add_argument('--demo_phi', default = None, type = int)
    parser.add_argument('--demo_phi_val', default = None)
    parser.add_argument('--neg_class', default = 0, type = int)
    parser.add_argument('--regularization', type = bool, default = 0)
    parser.add_argument('--max_label_flips', default = None, type=int)
    parser.add_argument('--regression', type = bool, default = 0)
    parser.add_argument('--tolerance', type = float, default = None)
    parser.add_argument('--find_lambda', type = bool, default = 0)
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
