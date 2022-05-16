import numpy as np
import copy

class Exact_Solver:
    def __init__(self, lin_reg, data_obj, target, args):
        self.lin_reg = lin_reg
        self.data_obj = data_obj
        self.label_perturb = args.label_per
        self.robust_radius = args.robust_rad
        self.target = target
        self.demo_phi = args.demo_phi
        self.demo_phi_val = args.demo_phi_val
        if args.max_label_flips == None:
            self.max_label_flips = 0.25 * self.lin_reg.x_train.shape[0]
        else:
            self.max_label_flips = args.max_label_flips

    def block_out_targeted_indices(self, matr_prod_x_abs):
        ''' If limiting Delta (i.e., label flipping) to only some points, make it so
            ineligible labels will not be flipped
        '''
        if self.target.target_val == -1:
            invalid_indices = np.where(np.transpose(self.lin_reg.x_train)[self.target.target_index] == 0)[1]
            matr_prod_x_abs[0,invalid_indices] = -1
        else:
            invalid_indices = np.where(np.transpose(self.lin_reg.x_train)[self.target.target_index] != self.target.target_val)[1]
            matr_prod_x_abs[0,invalid_indices] = -1
        return matr_prod_x_abs

    def find_perturbation_regression(self, x):
        orig_label = np.matmul(np.transpose(np.matmul(self.lin_reg.matr_prod, np.transpose(self.lin_reg.y_train))), np.transpose(x))[0,0]    
        matr_prod_x = np.matmul(x, self.lin_reg.matr_prod)
        matr_prod_x_abs = np.absolute(copy.deepcopy(matr_prod_x))

        if self.target.target_index is not None:
            matr_prod_x_abs = self.block_out_targeted_indices(matr_prod_x_abs)

        used_indices = []

        same = True
        iter = 0

        goal_label_max = orig_label + self.robust_radius
        goal_label_min = orig_label - self.robust_radius
        cur_label_min, cur_label_max = orig_label, orig_label

        while same == True and iter < self.max_label_flips:
            iter += 1 
            max_index = np.where(matr_prod_x_abs == (np.amax(matr_prod_x_abs)))[1][0]
            if matr_prod_x_abs[0,max_index] == -1:
                continue # edge case, flipped all labels
            
            matr_prod_x_abs[0,max_index] = -1
            used_indices.append(max_index)

            if matr_prod_x[0, max_index] > 0:
                self.y_mod_min[0,max_index] -= self.label_perturb
                self.y_mod_max[0,max_index] += self.label_perturb
                cur_label_min -= self.label_perturb * matr_prod_x[0,max_index]
                cur_label_max += self.label_perturb * matr_prod_x[0,max_index]
            else:
                self.y_mod_min[0,max_index] += self.label_perturb
                self.y_mod_max[0,max_index] -= self.label_perturb
                cur_label_min += self.label_perturb * matr_prod_x[0,max_index]
                cur_label_max -= self.label_perturb * matr_prod_x[0,max_index]
            
            if cur_label_min <= goal_label_min:
                same = False 
                self.success = True
                # double check our work
                if not np.matmul(np.transpose(np.matmul(self.lin_reg.matr_prod,np.transpose(self.y_mod_min))), np.transpose(x))[0,0] <= goal_label_min:
                    print("ERROR: actual val not satisfied (min)")
            elif cur_label_max >= goal_label_max:
                same = False
                self.success = True
                # sanity check
                if not np.matmul(np.transpose(np.matmul(self.lin_reg.matr_prod,np.transpose(self.y_mod_max))), np.transpose(x))[0,0] >= goal_label_max:
                    print("ERROR: actual val not satisfied (max)")
        self.perturbation_indices = used_indices
        if self.success:
            self.perturbation_size = len(used_indices)
        else:
            self.perturbation_size = -1
        
    def find_perturbation_cat(self, x):
        neg_class = self.data_obj.neg_class
        if neg_class == 0:
            threshold, multiplier = 0.5, 1
        else:
            threshold, multiplier = 0, 2
        if np.matmul(np.transpose(np.matmul(self.lin_reg.matr_prod, np.transpose(self.lin_reg.y_train))), np.transpose(x))[0,0] > threshold: 
            orig_label = 1
            goal_label = neg_class
        else:
            orig_label = neg_class 
            goal_label = 1
        cur_label = np.matmul(np.transpose(np.matmul(self.lin_reg.matr_prod, np.transpose(self.lin_reg.y_train))), np.transpose(x))[0,0]
        matr_prod_x = np.matmul(x, self.lin_reg.matr_prod)
        matr_prod_x_abs = np.absolute(copy.deepcopy(matr_prod_x))

        if self.target.target_index is not None:
            matr_prod_x_abs = self.block_out_targeted_indices(matr_prod_x_abs)

        same=True
        used_indices = []
        iter = 0
        while same == True and iter < self.lin_reg.y_train.shape[1]:
            iter+=1
            max_index = np.where(matr_prod_x_abs == (np.amax(matr_prod_x_abs)))[1][0]
            if matr_prod_x_abs[0,max_index] == -1:
                continue # edge case, flipped all labels
            
            matr_prod_x_abs[0,max_index] = -1

            if orig_label - goal_label > 0: # orig label 1, goal -1: want to decrease
                if matr_prod_x[0,max_index] > 0 and self.lin_reg.y_train[0,max_index] == 1: 
                    self.y_mod_min[0,max_index] = neg_class 
                    cur_label -= multiplier * matr_prod_x[0,max_index] 
                    used_indices.append(max_index) 
                elif matr_prod_x[0,max_index] < 0 and self.lin_reg.y_train[0,max_index] == neg_class: 
                    self.y_mod_min[0,max_index] = 1
                    cur_label += multiplier * matr_prod_x[0,max_index] 
                    used_indices.append(max_index)
            else: # orig label -1, goal 1
                if matr_prod_x[0,max_index] > 0 and self.lin_reg.y_train[0,max_index] == neg_class: 
                    self.y_mod_min[0,max_index] = 1
                    cur_label += multiplier * matr_prod_x[0,max_index] 
                    used_indices.append(max_index) 
                elif matr_prod_x[0,max_index] < 0 and self.lin_reg.y_train[0,max_index] == 1:
                    self.y_mod_min[0,max_index] = neg_class 
                    cur_label -= multiplier * matr_prod_x[0,max_index]
                    used_indices.append(max_index)
          
            if goal_label == 1 and cur_label > threshold: 
                same = False
                self.success = True
                # sanity check 
                if not np.matmul(np.transpose(np.matmul(self.lin_reg.matr_prod, np.transpose(self.y_mod_min))), np.transpose(x))[0,0] >= threshold:
                    print("ERROR: actual val not satisfied (trying to maximize label)")
            elif goal_label == neg_class and cur_label < threshold: 
                same = False
                self.success = True
                # sanity check 
                if not np.matmul(np.transpose(np.matmul(self.lin_reg.matr_prod, np.transpose(self.y_mod_min))), np.transpose(x))[0,0] <= threshold:
                    print("ERROR: actual val not satisfied (trying to minimize label)")

        self.perturbation_indices = used_indices
        if self.success:
            self.perturbation_size = len(used_indices)
        else:
            self.perturbation_size = -1

    def find_perturbation(self, start_index):
        self.success = False
        x = self.lin_reg.x_test[start_index]
        assert(self.lin_reg.matr_prod.shape[1] == self.lin_reg.y_train.shape[1])
        assert(self.lin_reg.matr_prod.shape[0] == x.shape[1])

        if self.demo_phi is not None:
            if float(self.demo_phi_val) == -1:
                if x[0,self.demo_phi] == 0:
                    return 0
            elif x[0,self.demo_phi] != float(self.demo_phi_val):
                return 0
       
        self.y_mod_min = copy.deepcopy(self.lin_reg.y_train)
        self.y_mod_max = copy.deepcopy(self.lin_reg.y_train)

        if self.data_obj.regression:
            self.find_perturbation_regression(x)
        else:
            self.find_perturbation_cat(x)
        return 1


