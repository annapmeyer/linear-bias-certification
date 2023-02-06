import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import dataset
import linear_eq
import perturbation
import argparse
import datetime

# silence UserWarnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
'''
Implements the approximate certification technique (Section 5 of the paper)

Required command-line parameters: dataset (name of the dataset) and datadir, 
TO DO fix with specifiying whole path
the directory where the train and test files are with names 
train_<dataset_name>_<fold> and test_<dataset_name>_<fold>. 
'''

def out_of_box(lin_reg,x_test,y_test):
    ''' Query the accuracy of the out-of-box logistic regression classifier'''
    clf = LogisticRegression(random_state=0)
    clf.fit(lin_reg.x_train, lin_reg.y_train)
    y_pred = clf.predict(x_test)
    return accuracy_score(y_test, y_pred)

def main(args):
    print(datetime.datetime.now())
    dataset_name = args.dataset
    metric = 'standard'
    label_col = 'label'

    target = perturbation.Target(args.target_index, args.target_val, target_dir=args.target_dir)
    data_obj = dataset.Dataset(dataset_name, args.data_train, args.data_test, label_col, target, args)
    args.orig_target_val = args.target_val
    args.orig_maj_group_val = args.maj_group_val
    args.orig_min_group_val = args.min_group_val
    args.orig_male_val = args.male_val
    args.orig_female_val = args.female_val

    if args.explain:
        thetas = []

    if args.l == -1 and (not args.accuracy):
        indices_orig = np.load(dataset_name + "_indices.npy").tolist()
    else:
        indices_orig = [args.l]

    if ('demo' in dataset_name) or ('synth' in dataset_name):
        num_folds = 4
    elif ('mnist' in dataset_name) or ('fitz' in dataset_name):
        num_folds = 1
    else:
        num_folds = 10
    if args.accuracy:
        all_acc, all_mse = [], []
    for i in range(num_folds):
        if num_folds == 1:
            x_train, y_train, x_test, y_test, scaler = data_obj.load_data(scale=args.scale)
        else:
            x_train, y_train, x_test, y_test, scaler = data_obj.load_data(i, scale=args.scale)

        if scaler is not None and (args.target_index is not None):
            # create an array that is all 0's except for target_val in target_index
            ary = np.zeros(x_test.shape[1])
            ary[args.target_index] = args.orig_target_val
            args.target_val = float(scaler.transform(ary.reshape(1, -1))[0,args.target_index])
        if scaler is not None:
            ary = np.zeros(x_test.shape[1])
            ary[args.maj_group_index] = args.orig_maj_group_val
            ary[args.gender_index] = args.orig_male_val
            transformed = scaler.transform(ary.reshape(1, -1))[0]

            args.maj_group_val = float(transformed[args.maj_group_index])
            args.male_val = float(transformed[args.gender_index])

            ary = np.zeros(x_test.shape[1])
            ary[args.gender_index] = args.orig_female_val
            ary[args.min_group_index] = args.orig_min_group_val
            transformed = scaler.transform(ary.reshape(1, -1))[0]
            args.min_group_val = float(transformed[args.min_group_index])
            args.female_val = float(transformed[args.gender_index])

        target = perturbation.Target(args.target_index, args.target_val, target_dir=args.target_dir)

        if args.ignore_indices is not None:
            # TO DO remove unwanted columns from x_train, x_test
            pass
        
        lin_reg = linear_eq.Linear_Regression(dataset_name, x_train, y_train, x_test, y_test, args)
        if args.accuracy:
            if args.regression:
                mse, acc = linear_eq.probe_accuracy_cont(lin_reg, x_test, y_test, args)
                all_acc.append(acc)
                all_mse.append(mse)
                print("fold ",i," accuracy: ",acc,", mse: ",mse)
            else:
                acc = linear_eq.probe_accuracy(lin_reg, x_test, y_test, args.neg_class, metric=metric)
                all_acc.append(acc)
                print("fold ",i," accuracy: ",acc)
            continue

        perturb_model = perturbation.Perturbation_Model(lin_reg, data_obj, target, args)
       
        if args.checkpoint:
            perturb_model.theta.load_theta(i, args.neg_class)
        
        indices = indices_orig.copy()
        total_lf = perturb_model.theta.get_perturbation_count()
        saturated=False
        label_flip_prev = 0
        while len(indices) > 0 and not saturated:
            while (len(indices) > 0 and total_lf >= indices[0]):                
                if len(indices) == 1:
                    indices = []
                else:
                    indices = indices[1:]

            if len(indices) > 0:
                labels_flip = indices[0] - total_lf
                perturb_model.perturb_theta(labels_flip)
                
                robust_pct_overall, robust_pct_men, robust_pct_women, robust_pct_maj, robust_pct_min  = \
                            perturb_model.eval_test_set()

                total_lf = perturb_model.theta.get_perturbation_count()
                if args.explain:
                    thetas.append(perturb_model.theta)
                    continue
                if total_lf == label_flip_prev:
                    saturated=True
                    continue

                
                label_flip_prev = total_lf

                if data_obj.regression: 
                    print("{args:",args,"}, {fold:",str(i), "}, {labels:",total_lf,"}, {robust:",robust_pct_overall,"}") 
                else:
                    print("{args:",args,"}, {fold:", str(i), "}, {labels:",total_lf, "}, {robust:",robust_pct_overall,
                    "}, {robust_pct_men: ",robust_pct_men,"}, {robust_pct_women:",robust_pct_women,"},{robust_pct_maj:",
                    robust_pct_maj,"}, {robust_pct_min:",robust_pct_min,"}")

                if robust_pct_overall == 0:
                    saturated = True  # silly to keep going if we're done

        perturb_model.theta.save_theta(i)
        break
    print(datetime.datetime.now())
    if args.accuracy:
        if args.regression:
            print('average mse: ',sum(all_mse)/len(all_mse))
        print("average acc: ",sum(all_acc)/len(all_acc))
    if args.explain:
        return thetas


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset')
    parser.add_argument('data_train') # full path to train data
    parser.add_argument('data_test') # full path to test data
    parser.add_argument('--l', default = -1, type = int) # specify label flips, otherwise, load from .npy file
    parser.add_argument('--checkpoint', type = bool, default = 0)
    parser.add_argument('--label_per', default = 0, type = float)
    parser.add_argument('--robust_rad', default = 0, type = float)
    parser.add_argument('--target_index', default = None, type = int)
    parser.add_argument('--target_val', default = None, type = float)
    parser.add_argument('--target_dir', default = 0, type = int)
    parser.add_argument('--neg_class', default = 0, type = int) # specify label of neg. class if not 0
    parser.add_argument('--regularization', type = bool, default = 1)
    parser.add_argument('--regression', type = bool, default = 0)
    parser.add_argument('--maj_group_index',default = None, type = int)
    parser.add_argument('--min_group_index',default = None, type = int)
    parser.add_argument('--gender_index',default = None, type = int)
    parser.add_argument('--maj_group_val', default = None)
    parser.add_argument('--min_group_val', default = None)
    parser.add_argument('--male_val', default = None)
    parser.add_argument('--female_val', default = None)
    parser.add_argument('--accuracy', default=False, type=bool)
    parser.add_argument('--tolerance', type=float, default=0) # accuracy-robustness tradeoff
    parser.add_argument('--explain', type = bool, default = 0) # return theta instead of the robustness rates
    parser.add_argument('--ignore_indices', type=int, default=None) # TO DO support list or some way of encoding multiple indices
    parser.add_argument('--scale',type=bool,default=False)
    args = parser.parse_args()
    args.find_lambda = False

    if args.target_index == -1:
        args.target_index = None
        args.target_val = None
    if args.tolerance == -1:
        args.regularization = False
        args.tolerance = 0

    if args.maj_group_index is not None:
        assert args.min_group_index is not None, "if you specific a majority group index you must also specify a minority group index"
        assert args.maj_group_val is not None, "if you specify a majority group index you must also specify a corresponding value"
        assert args.min_group_val is not None, "if you specify a minority group index you must also specify a corresponding value"
    if args.gender_index is not None:
        assert args.male_val is not None, "if you specify a gender index you must also specify a corresponding male value"
        assert args.female_val is not None, "if you specify a gender index you must also specify a corresponding female value"
    if (args.label_per != 0) or (args.robust_rad != 0):
        assert args.regression, "if you specify a label perturbation or robust radius, you must use regression"
    assert (args.label_per !=0) == (args.robust_rad != 0), "you must specify both label_per and robust_rad or neither"
    if args.target_index is not None:
        assert args.target_val is not None, "if you specify a target index you must specify a target value"
    main(args)

