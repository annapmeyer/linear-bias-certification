import numpy as np
import argparse
import linear_eq
import dataset
import perturbation

''' Find an appropriate lambda to use for a specified accuracy/robustness tradeoff '''


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset')
    parser.add_argument('datadir')
    parser.add_argument('--tolerance',type=float, default=0) # tradeoff
    parser.add_argument('--neg_class',type=int, default=0)
    parser.add_argument('--max_iter',type=int, default=10) # how many times to change and test lambda
    parser.add_argument('--regression', type=bool, default=0)
    parser.add_argument('--robust_rad', type=float, default=0)
    

    args = parser.parse_args()
    args.regularization = True
    args.find_lambda = True

    dataset_name = args.dataset
    label_col = 'label'
    args.tolerance = args.tolerance * 0.01 # convert pct to fraction

    train_filename = args.datadir + "/train_" + dataset_name + "_" 
    val_filename = args.datadir + "/val_" + dataset_name + "_"

    target = perturbation.Target(None, None) # include so Dataset doesn't break
    data_obj = dataset.Dataset(dataset_name, train_filename, val_filename, label_col, target, args)

    if ('compas' in dataset_name) or ('income' in dataset_name):
        num_folds = 10
    elif dataset_name == 'mnist':
        num_folds = 1
    else:
        num_folds = 4

    done = False
    baseline_acc, best_params = [], []


    lambdas = [0.01, 0.1, 1, 10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000, 1000000000]
    scores = []
    for l in lambdas:
        cur_scores = []
        for i in range(num_folds):
            x_train, y_train, x_val, y_val = data_obj.load_data(i)
            args.reg_val = l
            lin_reg = linear_eq.Linear_Regression(dataset_name,x_train, y_train, x_val, y_val, args)
            if args.regression:
                acc, otheracc = linear_eq.probe_accuracy_cont(lin_reg, x_val, y_val, args)
            else:
                acc = linear_eq.probe_accuracy(lin_reg, x_val, y_val, args.neg_class)
            cur_scores.append(acc)
        scores.append(sum(cur_scores)/len(cur_scores))
    if args.tolerance == 0:
        if args.regression:
            print("scores are ",scores)
            index_best = max(np.where(np.array(scores) == min(scores))[0])
        else:
            index_best = max(np.where(np.array(scores) == max(scores))[0])
    else: 
        if args.regression:
            index_best = max(np.where(np.array(scores) <= min(scores) + args.tolerance)[0])
            print("scores are ",scores, " and best score has to be at least ",min(scores)+args.tolerance)
            print("np where is ",np.where(np.array(scores) <= min(scores) + args.tolerance))
        else:
            print("scores are ",scores, " and best score has to be at least ",max(scores)-args.tolerance)
            print("np where is ",np.where(np.array(scores) >= max(scores) - args.tolerance))
            index_best = max(np.where(np.array(scores) >= max(scores) - args.tolerance)[0])
    best_l = lambdas[index_best]
    print("best lambda is ",best_l)
    if args.regression:
        min_score_part_one = min(scores)
    else:
        max_score_part_one = max(scores)

    upper,lower = lambdas[-1]*10, 0
    if index_best != 0:
        lower = lambdas[index_best-1]
    if index_best != len(lambdas) - 1:
        upper = lambdas[index_best+1]
    dist = upper - lower
    vals = [0, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2, 2.2, 2.4, 2.6, 2.8, 3, 3.2, 3.4, 3.6, 3.8, 4, 4.2, 4.4, 4.6, 4.8,
            5, 5.2, 5.4, 5.6, 5.8, 6, 6.2, 6.4,6.6, 6.8, 7, 7.2, 7.4, 7.6, 7.8, 8, 8.2, 8.4, 8.6, 8.8, 9]
    v_rev = [.9, .8, .7, .6, .5, .4, .3, .2, .1]
    scores = []
    for v in v_rev:
        test_l = best_l - v * best_l
        cur_scores = []
        for i in range(num_folds):
            args.reg_val = test_l
            x_train, y_train, x_val, y_val = data_obj.load_data(i)
            lin_reg = linear_eq.Linear_Regression(dataset_name, x_train, y_train, x_val, y_val, args)
            acc = linear_eq.probe_accuracy(lin_reg, x_val, y_val, args.neg_class)
            cur_scores.append(acc)
        scores.append(sum(cur_scores)/len(cur_scores))
    for v in vals:
        test_l = best_l + v * best_l
        cur_scores = []
        for i in range(num_folds):
            args.reg_val = test_l
            x_train, y_train, x_val, y_val = data_obj.load_data(i)
            lin_reg = linear_eq.Linear_Regression(dataset_name, x_train, y_train, x_val, y_val, args)
            acc = linear_eq.probe_accuracy(lin_reg, x_val, y_val, args.neg_class)
            cur_scores.append(acc)
        scores.append(sum(cur_scores)/len(cur_scores))
        if not args.regression:
            if sum(cur_scores) / len(cur_scores) < max(scores) - args.tolerance:
                break
        else:
            if sum(cur_scores) / len(cur_scores) > min(scores) + args.tolerance:
                break
    if args.tolerance == 0 and (not args.regression):
        index_best = max(np.where(np.array(scores) == max(scores))[0])
    elif args.tolerance == 0 and args.regression:
        index_best = max(np.where(np.array(scores) == min(scores))[0])
    elif not args.regression:
        index_best = max(np.where(np.array(scores) >= max(max(scores),max_score_part_one) - args.tolerance)[0])
    else:
        index_best = max(np.where(np.array(scores) <= min(min(scores),min_score_part_one) + args.tolerance)[0])
    if index_best < len(v_rev):
        best_l = best_l - v_rev[index_best]*best_l 
    else:
        best_l = best_l + vals[index_best-len(v_rev)]*best_l

    print("best lambda is ",best_l)
