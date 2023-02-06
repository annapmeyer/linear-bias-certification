import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler

# class copied from Rosenfeld et al.'s code (http://www.cs.cmu.edu/~elan/)
class Embedder(nn.Module):
    def __init__(self):
        super(Embedder, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 50)
        self.fc3 = nn.Linear(50, 8)

    def forward(self, x):
        return self.fc3(self.embed(x))

    def embed(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        return self.fc2(x)

def mnist_special(neg_class):
    network = Embedder()
    filename = 'chtc/embedder.pth'
    network.load_state_dict(torch.load(filename))
    network.eval()

    x_train = torch.tensor(np.load('chtc/xtrain_17.npy'))
    y_train = np.load('chtc/ytrain_17.npy')
    y_train[y_train == 1] = neg_class 
    y_train[y_train == 7] = 1

    x_test = torch.tensor(np.load('chtc/xtest_17.npy'))
    y_test = np.load('chtc/ytest_17.npy')
    y_test[y_test == 1] = neg_class 
    y_test[y_test == 7] = 1

    with torch.no_grad():
        train_embeddings = network.embed(x_train)
        test_embeddings = network.embed(x_test)
        np.save('chtc/xtrain_17_features.npy', train_embeddings)
        np.save('chtc/xtest_17_features.npy', test_embeddings)

    return np.matrix(train_embeddings), np.matrix(y_train), np.matrix(test_embeddings), np.matrix(y_test)

class Dataset:
    def __init__(self, name, train_filename, test_filename, label_col, target, args):        
        if target is not None and target.target_index is not None:
            self.name = name + "_t" + str(target.target_index) + "_v" + str(target.target_val)
        else:
            self.name = name
  
        self.train_filename = train_filename 
        self.test_filename = test_filename 
        self.regression = args.regression
        self.label_col = label_col
        self.neg_class = args.neg_class

    def add_intercept_column(self, x_train, x_test):
        b = pd.Series(1 for x in range(len(x_train)))
        x_train['b'] = b
        b2 = pd.Series(1 for x in range(len(x_test)))
        x_test['b'] = b2

    # assume that labels are coded either 0/1 or -1/1. Swap coding if necessary.
    def reconcile_neg_class(self, y_train, y_test):
        y_train_np = np.array(y_train)
        y_test_np = np.array(y_test)
        if not self.regression:
            if self.neg_class not in y_train.unique():
                if self.neg_class == 0:
                    y_train_np[y_train_np == -1] = 0
                    y_test_np[y_test_np == -1] = 0
                else:
                    y_train_np[y_train_np == 0] = -1
                    y_test_np[y_test_np == 0] = -1

        y_train = pd.Series(y_train_np)
        y_test = pd.Series(y_test_np)

    def load_data(self, fold = -1, scale = False):

        if self.name == "mnist":
            return mnist_special(self.neg_class)

        if '.npy' in self.train_filename:
            x_train = pd.DataFrame(np.load(self.train_filename))
            y_train = pd.Series(np.load(self.train_filename.split('.npy')[0] + '_labels.npy'))
            x_test = pd.DataFrame(np.load(self.test_filename))
            y_test = pd.Series(np.load(self.test_filename.split('.npy')[0] + '_labels.npy'))
        else:
            filename_train = self.train_filename if fold == -1 else self.train_filename + str(fold)
            filename_test = self.test_filename if fold == -1 else self.test_filename + str(fold)
            filename_train += ".csv"
            filename_test += ".csv"
            train = pd.read_csv(filename_train)
            test = pd.read_csv(filename_test)

            y_train = train[self.label_col]
            x_train = train.drop(columns=[self.label_col])
            y_test = test[self.label_col]
            x_test = test.drop(columns=[self.label_col])

        self.reconcile_neg_class(y_train, y_test)
        
        self.add_intercept_column(x_train, x_test)

        if scale:
            scaler = StandardScaler()
            x_train = pd.DataFrame(scaler.fit_transform(x_train))
            x_test = pd.DataFrame(scaler.transform(x_test))
        else:
            scaler = None
            
        return np.matrix(x_train), np.matrix(y_train).T, np.matrix(x_test), np.matrix(y_test).T, scaler