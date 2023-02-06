import numpy as np
from sklearn.metrics import f1_score

''' Helper functions to compute (X^T X)^{-1} X^T y.
    Also finds the value of lambda for the given dataset and tolerance. 
    Note that tolerance is the accuracy/robustness tradeoff, i.e., the maximal
    acceptable drop in accuracy in percentage points. Lambda can be computed with
    find_lambda.py. 

    theta = (x^T X - lambda I)^{-1} X^T y
'''

def compute_matr_prod(X, reg_val, ignore_indices=None):
    xT = np.transpose(X)
    if ignore_indices is not None:
        # TO DO remove the specified columns from X so that the (e.g.) sensitive attribute
        # is not included in the training
        pass
    if reg_val is not None:
        reg_prod = np.matmul(np.transpose(X), X) + reg_val * np.identity(X.shape[1])
        inv_prod = np.linalg.pinv(reg_prod)
    else:
        inv_prod = np.linalg.pinv(np.matmul(xT, X))
    return np.matmul(inv_prod, xT)

def probe_accuracy(lin_reg, x_test, y_test, neg_class, metric='standard'):
    theta = np.matmul(lin_reg.matr_prod, lin_reg.y_train)
    acc = 0

    if neg_class == 0:
        threshold = 0.5
    else:
        threshold = 0
    preds = np.matmul(x_test, theta)
    preds[preds > threshold] = 1
    preds[preds <= threshold] = neg_class

    if metric == 'f1':
        return f1_score(y_test, preds, average='binary')
    else:
        return np.sum(preds == y_test) / len(y_test)

def probe_accuracy_cont(lin_reg,x_test,y_test, args):
    theta = np.matmul(lin_reg.matr_prod, np.transpose(lin_reg.y_train))
    total_mse = 0

    for i in range(len(x_test)):
        total_mse += (y_test[0,i] - (np.matmul(x_test[i], theta))) ** 2
    mse = total_mse[0,0]/len(x_test)

    total_right = 0
    for i in range(len(x_test)):
        pred = np.matmul(x_test[i], theta)
        actual = y_test[0,i]
        if np.absolute(pred-actual) < args.robust_rad:
            total_right += 1
        
    pct_right = total_right/len(x_test)
    return mse, pct_right

def lookup_tolerance(dataset_name, tolerance):
    tol0 = {
        'mnist' : 0.042,
        'income' : 4.4, #  20,
        'incomega' : 22,
        'incomemd' : 1,
        'incomela' : 44,
        'incomeor' : 58,
        'compas' : 480,
        'compas_bal_whitenon': 480,
        'compas_bal_whiteblackall': 480,
        'compas_bal_whiteblack': 480,
        'demo' : 0.92,
        'heloc' : 0.9,
        'hmda' : 1000,
    }
    tol01 = {
        'income' : 1200, # 240,
        'compas' : 480,
        'mnist' : 0.18,
        'demo' : 2.2,
        'heloc' : 8.6,
        'hmda' :50000,# 1200,
    }
    tol02 = {
        'income' : 2000, # 240,
        'compas' : 520,
        'mnist' : 1.2,
        'demo' : 7.8,
        'heloc' : 54,
        'hmda': 100000,#2200,
    }
    tol05 = {
        'income' : 2800, # 700,
        'compas' : 900,
        'mnist' : 84,
        'demo' : 20,
        'heloc' : 460,
        'hmda' : 3200,
    }
    tol1 = {
        'mnist' : 1200,
        'income' : 3200, # 3400,
        'compas' : 1200,
        'demo' : 46,
        'heloc' : 2200,
        'hmda' : 10000000000,#4400,
    }
    tol15 = {
        'mnist' : 8000,
        'compas' : 1400,
        'income' : 3600, # 5400,
        'demo' : 64,
        'heloc' : 6600,
        'hmda' : 10000000000,#7000,
    }
    tol2 = {
        'mnist' : 14000,
        'compas': 2000,
        'income' : 7600, # 6000,
        'demo' : 120,
        'heloc': 16000,
        'hmda' : 10000000000,#10000,
        'incomega': 8000,
        'incomemd' : 9200,
        'incomeor' : 5200,
        'incomela' : 6000,
    }
    if tolerance == 0:
        return tol0.get(dataset_name, "dataset name not found")
    elif tolerance == 0.1:
        return tol01.get(dataset_name, "dataset name not found")
    elif tolerance == 0.2:
        return tol02.get(dataset_name, "dataset name not found")
    elif tolerance == 0.5:
        return tol05.get(dataset_name, "dataset name not found")
    elif tolerance == 1:
        return tol1.get(dataset_name, "dataset name not found")
    elif tolerance == 1.5:
        return tol15.get(dataset_name, "dataset name not found")
    elif tolerance == 2:
        return tol2.get(dataset_name, "dataset name not found")
    else:
        print("tolerance must be 0, 0.1, 0.2, or 0.5, 1, 1.5, or 2")
        return 0

class Linear_Regression:
    def __init__(self, dataset_name, x_train, y_train, x_test, y_test, args):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test =  y_test

        if dataset_name == 'income_reg':
            reg_val = 100
        elif args.find_lambda:
            reg_val = args.reg_val
        elif args.tolerance is not None:
            # if 'income' in dataset_name:
            #     dataset_name = 'income'
            reg_val = lookup_tolerance(dataset_name, args.tolerance)
            if reg_val == "dataset name not found":
                raise ValueError("dataset name not found")
        else:
            reg_val = None

        self.matr_prod = compute_matr_prod(self.x_train, reg_val)
