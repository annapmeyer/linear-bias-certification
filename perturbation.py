import linear_eq
import dataset
import copy
import numpy as np
import interval

''' Keep track of worst-case y perturbations and the resulting thetas '''

class Target:
    def __init__(self, target_index, target_val):
        self.target_index = target_index
        self.target_val = target_val
        self.specific_val = True
        if target_index is None:
            self.active = False
        else:
            self.active = True
            if target_val < 0:
                self.specific_val = False
        
class Theta_Perturbation:
    def __init__(self, d, lin_reg, data_name, target):
        self.dim = d
        self.data_name = data_name
        self.ymin_array = [copy.deepcopy(lin_reg.y_train) for i in range(d)]
        self.ymax_array = [copy.deepcopy(lin_reg.y_train) for i in range(d)]
        self.perturbed_indices_min = [[] for i in range(d)]
        self.perturbed_indices_max = [[] for i in range(d)]
        self.lin_reg = lin_reg
        self.upper = [0 for i in range(d)]
        self.lower = [0 for i in range(d)]
        self.row_copy_min = [np.absolute(copy.deepcopy(lin_reg.matr_prod[i])) for i in range(d)]
        self.row_copy_max = [np.absolute(copy.deepcopy(lin_reg.matr_prod[i])) for i in range(d)]

        if target.target_index is not None:
            self.block_out_targeted_indices(target)
        
        # if recover:
        #     self.load_theta()
            
    def block_out_targeted_indices(self, target):
        if target.target_val == -1:
            invalid_indices = np.where(np.transpose(self.lin_reg.x_train)[target.target_index] == 0)[1]
            for i in range(self.dim):
                self.row_copy_min[i][0, invalid_indices] = -1
                self.row_copy_max[i][0, invalid_indices] = -1
        else:
            invalid_indices = np.where(np.transpose(self.lin_reg.x_train)[target.target_index] != target.target_val)[1]
            for i in range(self.dim):
                self.row_copy_min[i][0, invalid_indices] = -1
                self.row_copy_max[i][0, invalid_indices] = -1

    def update_bounds(self, dim):
        self.lower[dim] = np.matmul(self.lin_reg.matr_prod[dim],np.transpose(self.ymin_array[dim]))[0]
        self.upper[dim] = np.matmul(self.lin_reg.matr_prod[dim], np.transpose(self.ymax_array[dim]))[0]

    def load_theta(self, fold, neg_class):
        for i in range(self.dim):
            self.perturbed_indices_min[i] = np.load("checkpoint_"+self.data_name + "_min_dim" + str(i) + "_fold" + str(fold) + ".npy").tolist()
            self.perturbed_indices_max[i] = np.load("checkpoint_"+self.data_name + "_max_dim" + str(i) + "_fold" + str(fold) + ".npy").tolist()

            for j in self.perturbed_indices_min[i]:
                self.row_copy_min[i][0,j] = -1
                if self.ymin_array[i][0,j] == neg_class:
                    self.ymin_array[i][0,j] = 1
                else:
                    self.ymin_array[i][0,j] = neg_class 
            for j in self.perturbed_indices_max[i]:
                self.row_copy_max[i][0,j] = -1
                if self.ymax_array[i][0,j] == neg_class: 
                    self.ymax_array[i][0,j] = 1
                else:
                    self.ymax_array[i][0,j] = neg_class
            
            self.update_bounds(i)
        
    def save_theta(self, fold):
        for i in range(self.dim):
            np.save("checkpoint_" + self.data_name + "_min_dim" + str(i) + "_fold" + str(fold) + ".npy", self.perturbed_indices_min[i])
            np.save("checkpoint_" + self.data_name + "_max_dim" + str(i) + "_fold" + str(fold) + ".npy", self.perturbed_indices_max[i])

    def get_perturbation_count(self):
        max_ymax = max(len(self.perturbed_indices_max[i]) for i in range(self.dim))
        max_ymin = max(len(self.perturbed_indices_min[i]) for i in range(self.dim))
        return max(max_ymax, max_ymin)

    def get_theta(self):
        return [interval.Interval(min(self.lower[i], self.upper[i]), 
                max(self.lower[i],self.upper[i])) for i in range(self.dim)]

class Perturbation_Model:
    def __init__(self, lin_reg:linear_eq.Linear_Regression, data_obj:dataset.Dataset, target, args):
        self.lin_reg = lin_reg
        self.data_obj = data_obj
        self.robust_rad = args.robust_rad
        self.label_per = args.label_per
        self.target = target
        self.gender_index = args.gender_index
        self.gender_val = args.gender_val
        self.maj_group_index = args.maj_group_index
        self.min_group_index = args.min_group_index
        self.maj_group_val = args.maj_group_val
        self.min_group_val = args.min_group_val
        self.neg_class = args.neg_class

        self.theta = Theta_Perturbation(len(self.lin_reg.matr_prod), lin_reg, data_obj.name, target)

    def perturb_theta_regression(self, labels_flip):
        ''' In the regression case, don't need to deal with min/max - instead just have single 
        perturbation set in perturbed_indices_min '''
        y = np.transpose(self.lin_reg.y_train)
        assert self.lin_reg.matr_prod.shape[1] == y.shape[0]
        d = self.theta.dim
        for i in range(d):
            for j in range(labels_flip):
                max_index = np.where(self.theta.row_copy_min[i] == (np.amax(self.theta.row_copy_min[i])))[1][0]
                if self.theta.row_copy_min[i][0,max_index] == -1: # edge case, flipped all labels
                    continue
                self.theta.row_copy_min[i][0,max_index] = -1 # -1 is fine, all other vals are pos
                self.theta.perturbed_indices_min[i].append(max_index)

                if self.lin_reg.matr_prod[i,max_index] >= 0:
                    self.theta.ymax_array[i][0,max_index] += self.label_per
                    self.theta.ymin_array[i][0,max_index] -= self.label_per
                else:
                    self.theta.ymax_array[i][0,max_index] -= self.label_per
                    self.theta.ymin_array[i][0,max_index] += self.label_per

    def perturb_theta_cat(self, labels_flip):
        d = self.theta.dim
        neg_class = self.neg_class

        for i in range(d):
            total_dec, total_incr, iter = 0, 0, 0
            while total_dec < labels_flip and iter<self.lin_reg.y_train.shape[1]:
                iter+=1
                max_index = np.where(self.theta.row_copy_min[i] == (np.amax(self.theta.row_copy_min[i])))[1][0]
                if self.theta.row_copy_min[i][0,max_index] == -1: # edge case, flipped all labels
                    iter=self.lin_reg.y_train.shape[1]
                    continue
                self.theta.row_copy_min[i][0,max_index] = -1 # all entries in row-copy are pos
                
                if self.lin_reg.matr_prod[i, max_index] >= 0 and self.theta.ymin_array[i][0,max_index] == 1:
                    self.theta.ymin_array[i][0,max_index] = neg_class
                    self.theta.perturbed_indices_min[i].append(max_index)
                    total_dec +=1  
                elif self.lin_reg.matr_prod[i, max_index] <= 0 and self.theta.ymin_array[i][0,max_index] == neg_class: 
                    self.theta.ymin_array[i][0,max_index] = 1
                    self.theta.perturbed_indices_min[i].append(max_index)
                    total_dec +=1 
            iter = 0 
            while total_incr < labels_flip and iter<self.lin_reg.y_train.shape[1]:
                iter += 1
                max_index = np.where(self.theta.row_copy_max[i] == (np.amax(self.theta.row_copy_max[i])))[1][0]
                if self.theta.row_copy_max[i][0,max_index] == -1: # edge case, flipped all labels
                    iter=self.lin_reg.y_train.shape[1]
                    continue
                self.theta.row_copy_max[i][0, max_index] = -1

                if self.lin_reg.matr_prod[i,max_index] >= 0 and self.theta.ymax_array[i][0, max_index] == neg_class: 
                    self.theta.ymax_array[i][0, max_index] = 1
                    self.theta.perturbed_indices_max[i].append(max_index)
                    total_incr += 1
                elif self.lin_reg.matr_prod[i,max_index] <= 0 and self.theta.ymax_array[i][0, max_index] == 1: 
                    self.theta.ymax_array[i][0, max_index] = neg_class
                    self.theta.perturbed_indices_max[i].append(max_index)
                    total_incr += 1

    def perturb_theta(self, labels_flip):
        if self.data_obj.regression:
            self.perturb_theta_regression(labels_flip)
        else:
            self.perturb_theta_cat(labels_flip)

        
        for i in range(self.theta.dim):
            self.theta.update_bounds(i)


    def eval_core_loop(self,n,theta,failure,demo_index,demo_val):
        robust, count = 0, 0
        for i in range(n):
            if demo_index is not None and (float(demo_val) == -1):
                if self.lin_reg.x_test[i][0,int(demo_index)] == 0:
                    continue
            elif demo_index is not None:
                if self.lin_reg.x_test[i][0, int(demo_index)] != float(demo_val):
                    continue
            count += 1
            result = self.eval_sample(theta, np.transpose(self.lin_reg.x_test[i]))
            if self.data_obj.regression:
                if result.upper - result.lower > self.robust_rad:
                    failure.append(i)
                else:
                    robust += 1
            else:    
                if self.neg_class == 0:
                    threshold = 0.5
                else:
                    threshold = 0
                if result.lower < threshold and result.upper > threshold: 
                    failure.append(i)
                else:
                    robust += 1
        if float(count) == 0.0:
            return -1, failure
        return float(robust)/float(count), failure

    def eval_test_set_overall(self):
        n = len(self.lin_reg.x_test)
        failure = []
        theta = self.theta.get_theta()
        robust, failure = self.eval_core_loop(n, theta, failure, None, None)
        return robust

    def eval_test_set_gender(self):
        if self.gender_index is not None:
            n = len(self.lin_reg.x_test)
            failure, theta = [], self.theta.get_theta()
            robust_men, failure = self.eval_core_loop(n, theta, failure, self.gender_index, self.gender_val)
            robust_women, failure = self.eval_core_loop(n, theta, failure, self.gender_index, 0)
            return robust_men, robust_women
        else:
            return -1, -1
    
    def eval_test_set_race(self):
        if self.maj_group_index is not None:
            n = len(self.lin_reg.x_test)
            failure, theta = [], self.theta.get_theta()
            robust_maj, failure = self.eval_core_loop(n, theta, failure, self.maj_group_index, self.maj_group_val)
            robust_min, failure = self.eval_core_loop(n, theta, failure, self.min_group_index, self.min_group_val)
            return robust_maj, robust_min
        else:
            return -1, -1

    def eval_test_set(self):
        n = len(self.lin_reg.x_test)
        robust_pct_overall = self.eval_test_set_overall()
        robust_pct_men, robust_pct_women, robust_pct_maj, robust_pct_min = -1, -1, -1, -1
        if self.gender_index is not None:
            robust_pct_men, robust_pct_women = self.eval_test_set_gender()
        if self.maj_group_index is not None:
            robust_pct_maj, robust_pct_min = self.eval_test_set_race()
        
        return robust_pct_overall, robust_pct_men, robust_pct_women, robust_pct_maj, robust_pct_min

    def eval_sample(self, theta, x):
        ''' Perform matrix multiplication when theta (really theta^T) is intervals
            theta is a len-d array and x is d x 1'''
        assert x.shape[1] == 1
        assert len(theta) == x.shape[0]

        total = interval.Interval(0, 0)
        for i in range(len(theta)):
            total = interval.int_add(total, interval.int_mult(theta[i], interval.Interval(x[i,0],x[i,0])))
        return total

        



                


        