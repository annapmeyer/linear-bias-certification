import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import dataset
import linear_eq
import perturbation
import argparse

'''
Implements the approximate certification technique (Section 5 of the paper)

Required command-line parameters: dataset (name of the dataset) and datadir, 
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
    dataset_name = args.dataset
    label_col = 'label'

    train_filename = args.datadir + "/train_" + dataset_name + "_"
    test_filename =  args.datadir + "/test_" + dataset_name + "_"

    target = perturbation.Target(args.target_index, args.target_val)
    data_obj = dataset.Dataset(dataset_name, train_filename, test_filename, label_col, target, args)


    if args.l == -1 and (not args.accuracy):
        indices_orig = np.load("np_indices/" + dataset_name + "_indices.npy").tolist()[0]
    else:
        indices_orig = [args.l]

    if ('demo' in dataset_name) or ('synth' in dataset_name):
        num_folds = 4
    elif 'mnist' in dataset_name:
        num_folds = 1
    else:
        num_folds = 10
    if args.accuracy:
        all_acc, all_mse = [], []
    for i in range(num_folds):
        x_train, y_train, x_test, y_test = data_obj.load_data(i)
        
        lin_reg = linear_eq.Linear_Regression(dataset_name, x_train, y_train, x_test, y_test, args)
        if args.accuracy:
            if args.regression:
                mse, acc = linear_eq.probe_accuracy_cont(lin_reg, x_test, y_test, args)
                all_acc.append(acc)
                all_mse.append(mse)
                print("fold ",i," accuracy: ",acc,", mse: ",mse)
            else:
                acc = linear_eq.probe_accuracy(lin_reg, x_test, y_test, args.neg_class)
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
    if args.accuracy:
        if args.regression:
            print('average mse: ',sum(all_mse)/len(all_mse))
        print("average acc: ",sum(all_acc)/len(all_acc))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset')
    parser.add_argument('datadir')
    parser.add_argument('--l', default = -1, type = int) # specify label flips, otherwise, load from .npy file
    parser.add_argument('--checkpoint', type = bool, default = 0)
    parser.add_argument('--label_per', default = 0, type = float)
    parser.add_argument('--robust_rad', default = 0, type = float)
    parser.add_argument('--target_index', default = None, type = int)
    parser.add_argument('--target_val', default = None, type = float)
    parser.add_argument('--neg_class', default = 0, type = int) # specify label of neg. class if not 0
    parser.add_argument('--regularization', type = bool, default = 1)
    parser.add_argument('--regression', type = bool, default = 0)
    parser.add_argument('--maj_group_index',default = None, type = int)
    parser.add_argument('--min_group_index',default = None, type = int)
    parser.add_argument('--gender_index',default = None, type = int)
    parser.add_argument('--maj_group_val', default = None)
    parser.add_argument('--min_group_val', default = None)
    parser.add_argument('--gender_val', default = None)
    parser.add_argument('--accuracy', default=False, type=bool)
    parser.add_argument('--tolerance', type=float, default=0) # accuracy-robustness tradeoff
    
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
        assert args.gender_val is not None, "if you specify a gender index you must also specify a corresponding value"
    if (args.label_per != 0) or (args.robust_rad != 0):
        assert args.regression, "if you specify a label perturbation or robust radius, you must use regression"
    assert (args.label_per !=0) == (args.robust_rad != 0), "you must specify both label_per and robust_rad or neither"
    if args.target_index is not None:
        assert args.target_val is not None, "if you specify a target index you must specify a target value"
    main(args)

