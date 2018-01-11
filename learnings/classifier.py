"""
Using Scikit library to build all types of classifiers & regression
1.1 Generalized Linear Model (Partial)
1.2 Linear and Quadratic Discriminant Analysis
1.3 Kernel ridge regression
1.4 Support Vector Machines
1.5 Stochastic Gradient Descent
1.6 Nearest Neighbors
1.7 Gaussian Processes
1.8 Cross decomposition
1.9 Naive Bayes (Completed)
1.10 Decision Trees (Completed)
1.11 Ensemble methods
1.12 Multiclass and multilabel algorithm
1.13 Feature Selection


Ref: H. Zhang (2004). The optimality of Naive Bayes. Proc. FLAIRS.
"""

from __future__ import division;
from __future__ import print_function;
from __future__ import absolute_import;


import os;
import argparse;
import logging;
import numpy as np;

def main(*args, **kwargs):
    """
    """
    logging.basicConfig(level=logging.INFO);
    logger = logging.getLogger('Classifier');

    parser = argparse.ArgumentParser( \
        prog = 'Sckit Classifier', \
        description='Scikit Classifier Program', \
        add_help = True);
    # Input information
    parser.add_argument('--input_type', \
        dest = 'input_type', \
        choices = ['demo'], \
        default = 'demo', \
        help = 'Input data type');
    parser.add_argument('--test_size', \
        dest = 'test_size', \
        default = 0.2, \
        help = 'Percentage of data are used for validation');
    parser.add_argument('--mode', \
        dest = 'mode', \
        choices = ['train', 'predict', 'train_predict'], \
        default = 'train_predict', \
        required = True, \
        help = 'Different functions of the program');
    parser.add_argument('--model', \
        dest = 'model', \
        choices  = [ 'linear_model', 'LinearRegression', 'Ridge', 'RidgeCV', \
            'gaussian_process', 'GaussianProcessRegressor', 'GaussianProcessClassifier', \
            'tree', 'DecisionTreeRegressor', 'DecisionTreeClassifier', \
            'naive_bayes', 'GaussianNB', 'MultinomialNB','BernoulliNB'], \
        action = 'store', \
        default = 'naive_bayes', \
        help = 'Type of classifier/regression models');
    parser.add_argument('--model_file', \
        dest = 'model_file', \
        action = 'store', \
        default = 'model.pkl', \
        help = 'Load the classifier/regression model for prediction');
    parser.add_argument('--output_dir', \
        dest = 'output_dir', \
        action = 'store', \
        default = './work', \
        help = 'Location of the output folder');
    parser.add_argument('--doc', \
        dest = 'documents', \
        action = 'store_const', \
        const = '1', \
        default = 0, \
        help = 'Print information from the program');

    args = parser.parse_args();
    if args.documents:
        print_help();

    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir);
    
    data = Dataset(mode = args.input_type, test_size = args.test_size, output_dir = args.output_dir);
    obj = Model(model = args.model, dataset = data, output_dir = args.output_dir);
    if args.mode == 'train':
        obj.train();
        obj.save(args.model_file);
    elif args.mode == 'predict':
        obj.load(args.model_file);
        obj.predict();
    elif args.mode == 'train_predict':
        obj.train();
        obj.save(args.model_file);
        obj.predict();
    return;

class Dataset(object):
    """
    """
    def __init__(self, mode, test_size, output_dir):
        """
        TODO: Plot data
        """
        self.logger = logging.getLogger('Dataset');
        self.logger.info('Initialize dataset');
        self.output_dir = output_dir;
        self.test_size = test_size;
        if mode == 'demo':
            self.length = 100000;
            self.demo(length = self.length, test_size = self.test_size);
        return;

    def randomize_dataset(self, x, y, test_size):
        """
        Randomize dataset 
        """
        N = x.shape;
        ind_cut = int(test_size * N[0]);
        ind = np.random.permutation(N[0]);
        #X_train, X_test, Y_train, Y_test
        return x[ind[ind_cut:]], x[ind[:ind_cut]], y[ind[ind_cut:]], y[ind[:ind_cut]]

    def demo(self, length, test_size):
        """
        Demo dataset
        """
        self.logger.info('Load demo dataset');
        noise = np.random.uniform(-2, 2, length);
        x = np.array([range(length),np.random.uniform(-10, 10, length)]);
        x = np.transpose(x);
        period = np.random.randint(length/2);
        y = np.sin(np.pi * x[:,1] / period / 2) + x[:,1] / period + noise;
        y = y.astype(int);
        self.x_train, self.x_test, self.y_train, self.y_test = self.randomize_dataset(x, y, test_size);
        return;

class Model(object):
    """
    """
    def __init__(self, model, dataset, output_dir):
        """
        """
        self.logger = logging.getLogger('Model');
        self.logger.info('Initialize classifier/regression model');
        self.output_dir = output_dir;
        self.model = model;
        self.x_train = dataset.x_train;
        self.y_train = dataset.y_train;
        self.x_test = dataset.x_test;
        self.y_test = dataset.y_test;
        return;

    def save(self, model_file):
        """
        """
        self.logger.info('Saving model');
        from sklearn.externals import joblib;
        joblib.dump(self.clf, model_file);
        return model_file;

    def load(self):
        """
        """
        self.logger.info('Loading model');
        from sklearn.externals import joblib;
        joblib.dump(self.clf, model_file); 
        return self.clf;

    def train(self):
        """
        """
        if self.model == 'Ridge':
            self.Ridge(x = self.x_train, y = self.y_train);
        elif self.model == 'RidgeCV':
            self.RidgeCV(x = self.x_train, y = self.y_train);
        elif self.model == 'LinearRegression':
            self.LinearRegression(x = self.x_train, y = self.y_train);
        elif self.model == 'DecisionTreeClassifier':
            self.DecisionTreeClassifier(x = self.x_train, y = self.y_train, \
                output_dir = self.output_dir);
        elif self.model == 'DecisionTreeRegressor':
             self.DecisionTreeRegressor(x = self.x_train, y = self.y_train);
        elif self.model == 'GaussianProcessClassifier':
            self.GaussianProcessClassifier(x = self.x_train, y = self.y_train);
        elif self.model == 'GaussianProcessRegressor':
            self.GaussianProcessRegressor(x = self.x_train, y = self.y_train);
        elif self.model == 'GaussianNB':
            self.GaussianNB(x = self.x_train, y = self.y_train);
        elif self.model == 'MultinomialNB':
            self.MultinomialNB(x = self.x_train, y = self.y_train);
        elif self.model == 'BernoulliNB':
            self.BernoulliNB(x = self.x_train, y = self.y_train);
        self.get_score('Training', x = self.x_train, y = self.y_train);
        return;


    def LinearRegression(self, x, y):
        """
        Linear Regression:
        Ordinary Least Squares
        LinearRegression will take in its fit method arrays X, y and will store
        the coefficients w of the linear model in its coef_ member.
        """
        self.logger.info('Perform Linear Regression');
        from sklearn import linear_model;
        self.clf = linear_model.LinearRegression();
        self.clf.fit(x, y);
        return self.clf;


    def Ridge(self, x, y):
        """
        Linear Regression:
        Ridge regression addresses some of the problems of Ordinary Least Squares 
        by imposing a penalty on the size of coefficients. The ridge coefficients 
        minimize a penalized residual sum of squares.
        """
        self.logger.info('Perform Ridge Regression');
        from sklearn import linear_model;
        self.clf = linear_model.Ridge(alpha = .5);
        self.clf.fit(x, y);
        return self.clf;

    def RidgeCV(self, x, y):
        """
        Linear Regression:
        RidgeCV implements ridge regression with built-in cross-validation of 
        the alpha parameter. The object works in the same way as GridSearchCV 
        except that it defaults to Generalized Cross-Validation (GCV), 
        an efficient form of leave-one-out cross-validation.
        """
        self.logger.info('Perform RidgeCV Regression');
        from sklearn import linear_model;
        self.clf = linear_model.RidgeCV(alphas=[0.1, 1.0, 10.0]);
        self.clf.fit(x, y);
        return self.clf;


    def Lasso(self, x, y):
        """
        Linear Regression:
        The Lasso is a linear model that estimates sparse coefficients. 
        It is useful in some contexts due to its tendency to prefer solutions 
        with fewer parameter values, effectively reducing the number of variables 
        upon which the given solution is dependent. 
        """
        self.logger.info('Perform Lasso Regression');
        from sklearn import linear_model;
        self.clf = linear_model.Lasso(alpha = 0.1);
        self.clf.fit(x, y);
        return self.clf;


    def GaussianProcessRegressor(self, x, y):
        """
        The GaussianProcessRegressor implements Gaussian processes (GP) for regression purposes. 
        For this, the prior of the GP needs to be specified. The prior mean is assumed to be 
        constant and zero (for normalize_y=False) or the training data’s mean (for normalize_y=True). 
        The prior’s covariance is specified by a passing a kernel object. The hyperparameters of the 
        kernel are optimized during fitting of GaussianProcessRegressor by maximizing the 
        log-marginal-likelihood (LML) based on the passed optimizer. As the LML may have multiple local 
        optima, the optimizer can be started repeatedly by specifying n_restarts_optimizer. The first 
        run is always conducted starting from the initial hyperparameter values of the kernel; subsequent 
        runs are conducted from hyperparameter values that have been chosen randomly from the range of 
        allowed values. If the initial hyperparameters should be kept fixed, None can be passed as optimizer.
        """
    
        return self.clf;

    def GaussianProcessClassifier(self, x, y):
        """
        The GaussianProcessClassifier implements Gaussian processes (GP) for 
        classification purposes, more specifically for probabilistic classification, 
        where test predictions take the form of class probabilities. GaussianProcessClassifier 
        places a GP prior on a latent function f, which is then squashed through a link function 
        to obtain the probabilistic classification. The latent function f is a so-called nuisance 
        function, whose values are not observed and are not relevant by themselves. Its purpose 
        is to allow a convenient formulation of the model, and f is removed (integrated out) 
        during prediction. GaussianProcessClassifier implements the logistic link function, for 
        which the integral cannot be computed analytically but is easily approximated in the binary case.
        """


    def DecisionTreeClassifier(self, x, y, output_dir):
        """
        Tree Classifier
        DecisionTreeClassifier is a class capable of performing multi-class
        classification on a dataset.
        """
        self.logger.info('Perform Decision Tree Classifier');
        from sklearn import tree;
        self.clf = tree.DecisionTreeClassifier();
        self.clf.fit(x, y);
        
        # Save the tree
        #import graphviz;
        #dot_data = tree.export_graphviz(self.clf, out_file=None);
        #graph = graphviz.Source(dot_data);

        return self.clf;


    def DecisionTreeRegressor(self, x, y):
        """
        Tree Regression
        Decision trees can also be applied to regression problems.
        """
        self.logger.info('Perform Decision Tree Classifier');
        from sklearn import tree;
        self.clf = tree.DecisionTreeRegressor();
        self.clf.fit(x, y);
        return self.clf;      


    def BernoulliNB(self, x, y):
        """
        Naive Bayes Classifier:
        BernoulliNB implements the naive Bayes training and classification algorithms 
        for data that is distributed according to multivariate Bernoulli distributions
        """
        self.logger.info('Perform BernoulliNB training');
        from sklearn.naive_bayes import BernoulliNB;
        self.clf = BernoulliNB();
        self.clf.fit(x, y);
        return self.clf;


    def MultinomialNB(self, x, y):
        """
        Naive Bayes Classifier:
        MultinomialNB implements the naive Bayes algorithm for multinomially distributed data.
        Bayes variants used in text classification (where the data are typically represented 
        as word vector counts, although tf-idf vectors are also known to work well in practice).
        """
        self.logger.info('Perform MultinomialNB training');
        from sklearn.naive_bayes import MultinomialNB;
        self.clf = MultinomialNB();
        self.clf.fit(x,y);
        return self.clf;


    def GaussianNB(self, x, y):
        """
        Naive Bayes Classifier:
        GaussianNB implements the Gaussian Naive Bayes algorithm for classification. 
        The likelihood of the features is assumed to be Gaussian.
        """
        self.logger.info('Perform GaussianNB training');
        from sklearn.naive_bayes import GaussianNB;
        self.clf = GaussianNB();
        self.clf.fit(x,y);
        return self.clf;


    def get_score(self, desc, x, y):
        """
        Print the coefficient of determination R^2 of the prediction.
        """
        self.score = self.clf.score(x, y);
        self.logger.info(str(desc) + ' Score(R^2): ' + str(self.score));
        return self.score;

    def get_loss(self, y_ref, y_pred):
        """
        Compute the differences between the predicted score and actual score
        """
        self.loss = 0;
        for a,b in zip(y_ref, y_pred):
            self.loss += a - b;
        self.logger.info('Total Loss: ' + str(self.loss));
        return self.loss;

    def predict(self):
        """
        Perform regression or classification of the test data
        TODO: Add save function to save the data
        """
        self.logger.info('Input: ' + str(self.x_test));
        self.predict = self.clf.predict(self.x_test);
        self.logger.info('Prediction: ' + str(self.predict));
        self.get_score('Actual', x = self.x_test, y = self.y_test);
        self.get_score('Predict', x = self.x_test, y = self.predict);
        self.get_loss(y_ref = self.y_test, y_pred = self.predict);

            
        return self.predict;

def print_help():
    """
    Print document
    """
    print(__doc__);
    exit();
    return;


if __name__ == '__main__':
    main();

