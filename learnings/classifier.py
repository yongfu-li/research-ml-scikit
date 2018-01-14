"""
Using Scikit library to build all types of classifiers & regression


Ref: H. Zhang (2004). The optimality of Naive Bayes. Proc. FLAIRS.
"""

from __future__ import division;
from __future__ import print_function;
from __future__ import absolute_import;

import os;
import argparse;
import logging;
import numpy as np;
from tqdm import tqdm;
import dill;
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

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
        help = 'Percentage of data used for validation');
    parser.add_argument('--mode', \
        dest = 'mode', \
        choices = ['train', 'test', 'train_test'], \
        default = 'train_test', \
        required = True, \
        help = 'Different functions of the program');
    parser.add_argument('--model', \
        dest = 'model', \
        choices  = [ 'linear_model', 'LinearRegression', 'Ridge', 'RidgeCV', \
            'support_vector_machine', 'SVC', 'NuSVC', 'LinearSVC', \
            'stochastic_gradient_descent', 'SGDClassifier', \
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
        obj.save_model(os.path.join(args.output_dir,'model.pkl'));
        obj.plot(mode = 'train', model = args.model, output_dir = args.output_dir);
    elif args.mode == 'test':
        obj.load_model(args.model_file);
        obj.predict();
        obj.plot(mode = 'test', model = args.model, output_dir = args.output_dir);
    elif args.mode == 'train_test':
        obj.train();
        obj.save_model(os.path.join(args.output_dir,'model.pkl'));
        obj.predict();
        obj.plot(mode = 'train', model = args.model, output_dir = args.output_dir);
        obj.plot(mode = 'test', model = args.model, output_dir = args.output_dir);
    output_file_setting = os.path.join(args.output_dir, 'setting.pkl');
    save_object(obj, output_file_setting);
    return;

def save_object(obj, filename):
    """
    """
    with open(filename, 'wb') as fw:
        dill.dump(obj, fw);

def load_object(filename):
    """
    """
    with open(filename, 'rb') as fp:
        obj = dill.load(fp);
    return obj;

class Dataset(object):
    """
    """
    def __init__(self, mode, test_size = 0.2, output_dir = './work'):
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
    1.1 Generalized Linear Model (Partial)
    1.2 Linear and Quadratic Discriminant Analysis
    1.3 Kernel ridge regression
    1.4 Support Vector Machines 
        - SVC
        - NuSVC
        - LinearSVC
    1.5 Stochastic Gradient Descent
        - SGDClassifier
    1.6 Nearest Neighbors
        - KNeighborsRegressor
        - RadiusNeighborsRegressor
    1.7 Gaussian Processes 
        - GaussianProcessClassifier
        - GaussianProcessRegressor
    1.8 Cross decomposition
    1.9 Naive Bayes
        - GaussianNB
        - MultinomialNB
        - BernoulliNB
    1.10 Decision Trees 
        - DecisionTreeClassifier
        - DecisionTreeRegressor 
    1.11 Ensemble methods
    1.12 Multiclass and multilabel algorithm
    1.13 Feature Selection
    """
    
    def __init__(self, model, dataset, output_dir = './work'):
        """
        """
        self.logger = logging.getLogger('Model-Input');
        self.logger.info('Initialize classifier/regression model');
        self.output_dir = output_dir;
        self.model = model;
        self.x_train = dataset.x_train;
        self.y_train = dataset.y_train;
        self.x_test = dataset.x_test;
        self.y_test = dataset.y_test;
        return;

    def train(self):
        """
        """
        self.logger = logging.getLogger('Model-Training');
        if self.model == 'Ridge':
            self.Ridge(x = self.x_train, y = self.y_train);
        elif self.model == 'RidgeCV':
            self.RidgeCV(x = self.x_train, y = self.y_train);
        elif self.model == 'LinearRegression':
            self.LinearRegression(x = self.x_train, y = self.y_train);
        elif self.model == 'Lasso':
            self.Lasso(x = self.x_train, y = self.y_train);
        elif self.model == 'SVC':
            self.SVC(x = self.x_train, y = self.y_train);
        elif self.model == 'NuSVC':
            self.NuSVC(x = self.x_train, y = self.y_train);
        elif self.model == 'LinearSVC':
            self.LinearSVC(x = self.x_train, y = self.y_train);
        elif self.model == 'SGDClassifier':
            self.SGDClassifier(x = self.x_train, y = self.y_train);
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
        return;

    def LinearRegression(self, x, y):
        """
        Linear Regression:
        Ordinary Least Squares
        LinearRegression will take in its fit method arrays X, y and will store
        the coefficients w of the linear model in its coef_ member.

        sklearn.linear_model.LinearRegression(fit_intercept=True, normalize=False, copy_X=True, n_jobs=1)
        """
        self.logger.info('Perform Linear Regression');
        from sklearn import linear_model;
        self.type = 'regression';
        self.clf = linear_model.LinearRegression(n_jobs = -1);
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
        self.type = 'regression';        
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
        self.type = 'regression';
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
        self.type = 'regression';
        self.clf = linear_model.Lasso(alpha = 0.1);
        self.clf.fit(x, y);
        return self.clf;

    
    def SVC(self, x, y):
        """
        Support Vector Machine Classifier
        Support Vector Machine for Regression implemented using libsvm.
        """
        self.logger.info('Perform SVC Classifier');
        from sklearn.svm import SVC;
        self.type = 'classification';
        self.clf = SVC();
        self.clf.fit(x, y);
        return self.clf;


    def NuSVC(self, x, y):
        """
        Nu-Support Vector Machine Classifier
        Similar to SVC but uses a parameter to control the number of support vectors.
        """
        self.logger.info('Perform NuSVC Classifier');
        from sklearn.svm import NuSVC;
        self.type = 'classification';
        self.clf = NuSVC();
        self.clf.fit(x, y);
        return self.clf;


    def LinearSVC(self, x, y):
        """
        Linear-Support Vector Machine Classifier
        Similar to SVC with parameter kernel=’linear’, but implemented in terms of liblinear 
        rather than libsvm, so it has more flexibility in the choice of penalties and loss 
        functions and should scale better to large numbers of samples.
        """
        self.logger.info('Perform LinearSVC Classifier');
        from sklearn.svm import LinearSVC;
        self.type = 'classification';
        self.clf = LinearSVC();
        self.clf.fit(x, y);
        return self.clf;


    def SGDClassifier(self, x, y):
        """
        The class SGDClassifier implements a plain stochastic gradient descent learning routine 
        which supports different loss functions and penalties for classification.
        class sklearn.linear_model.SGDClassifier(loss=’hinge’, penalty=’l2’, alpha=0.0001, 
        l1_ratio=0.15, fit_intercept=True, max_iter=None, tol=None, shuffle=True, verbose=0, 
        epsilon=0.1, n_jobs=1, random_state=None, learning_rate=’optimal’, eta0=0.0, power_t=0.5, 
        class_weight=None, warm_start=False, average=False, n_iter=None)

        ‘hinge’, ‘log’, ‘modified_huber’, ‘squared_hinge’, ‘perceptron’, or 
        a regression loss: ‘squared_loss’, ‘huber’, ‘epsilon_insensitive’, or ‘squared_epsilon_insensitive’.
        """

        self.logger.info('Perform SGDClassifier Classifier');
        from sklearn.linear_model import SGDClassifier;
        self.type = 'classification';
        self.clf = SGDClassifier();
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
        self.type = 'classification';
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
        self.logger.info('Perform Decision Tree Regression');
        from sklearn import tree;
        self.type = 'regression';
        self.clf = tree.DecisionTreeRegressor();
        self.clf.fit(x, y);
        return self.clf;      


    def BernoulliNB(self, x, y):
        """
        Naive Bayes Classifier:
        BernoulliNB implements the naive Bayes training and classification algorithms 
        for data that is distributed according to multivariate Bernoulli distributions
        """
        self.logger.info('Perform BernoulliNB Classifier');
        from sklearn.naive_bayes import BernoulliNB;
        self.type = 'classification';
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
        self.logger.info('Perform MultinomialNB Classifier');
        from sklearn.naive_bayes import MultinomialNB;
        self.type = 'classification';
        self.clf = MultinomialNB();
        self.clf.fit(x,y);
        return self.clf;


    def GaussianNB(self, x, y):
        """
        Naive Bayes Classifier:
        GaussianNB implements the Gaussian Naive Bayes algorithm for classification. 
        The likelihood of the features is assumed to be Gaussian.
        """
        self.logger.info('Perform GaussianNB Classifier');
        from sklearn.naive_bayes import GaussianNB;
        self.type = 'classification';
        self.clf = GaussianNB();
        self.clf.fit(x,y);
        return self.clf;


    def get_score(self, x, y):
        """
        Print the coefficient of determination R^2 of the prediction.
        """
        self.logger = logging.getLogger('Model-Accuracy');        
        self.score = self.clf.score(x, y);
        self.logger.info('Score (R^2 - coefficient of determination): ' + str(self.score));        
        return self.score;


    def explained_variance_score(self, y_ref, y_pred):
        """
        Explained variance regression score function
        Best possible score is 1.0, lower values are worse.
        """
        import sklearn.metrics as metrics;
        self.evs_value = metrics.explained_variance_score( \
            y_true = y_ref, \
            y_pred = y_pred, \
            multioutput = 'uniform_average');
        self.logger.info('Explained Variance Regression (Max 1.0): ' + str(self.evs_value));
        return self.evs_value;


    def mean_absolute_error(self, y_ref, y_pred):
        """
        Mean absolute error regression loss
        """
        import sklearn.metrics as metrics;
        self.mae_value = metrics.mean_absolute_error( \
            y_true = y_ref, \
            y_pred = y_pred, \
            sample_weight = None, \
            multioutput = 'uniform_average');
        self.logger.info('Mean Absolute Error: ' + str(self.mae_value));
        return self.mae_value;


    def mean_squared_error(self, y_ref, y_pred):
        """
        Mean squared error regression loss
        """
        import sklearn.metrics as metrics;
        self.mse_value = metrics.mean_squared_error( \
            y_true = y_ref, \
            y_pred = y_pred, \
            sample_weight = None, \
            multioutput = 'uniform_average');
        self.logger.info('Mean Squared Error (L2 Loss): ' + str(self.mse_value));
        return self.mse_value;


    def mean_squared_log_error(self, y_ref, y_pred):
        """
        Mean squared logarithmic error regression loss
        """
        import sklearn.metrics as metrics;
        self.msle_value = -1;
        try:
            self.msle_value = metrics.mean_squared_log_error( \
                y_true = y_ref, \
                y_pred = y_pred, \
                sample_weight = None, \
                multioutput = 'uniform_average');
            self.logger.info('Mean Squared Logarithmic Error: ' + str(self.msle_value));
        except:
            self.logger.info('Mean Squared Logarithmic Error: Negative Value Present');
        return self.msle_value;


    def median_absolute_error(self, y_ref, y_pred):
        """
        Median absolute error regression loss
        """
        import sklearn.metrics as metrics;
        self.mnse_value = metrics.median_absolute_error( \
            y_true = y_ref, \
            y_pred = y_pred);
        self.logger.info('Median Absolute Error: ' + str(self.mnse_value));
        return self.mnse_value;


    def total_loss(self, y_ref, y_pred):
        """
        Total Loss
        """
        self.total_loss_value = 0;
        for a,b in zip(y_ref, y_pred):
            self.total_loss_value += abs(a - b);
        self.logger.info('Total Loss (L1 Loss): ' + str(self.total_loss_value));

        return self.total_loss_value;

    def r2_score(self, y_ref, y_pred):
        """
        R^2 (coefficient of determination) regression score function.
        Best possible score is 1.0 and it can be negative (because the model can 
        be arbitrarily worse). A constant model that always predicts the expected 
        value of y, disregarding the input features, would get a R^2 score of 0.0.
        """
        import sklearn.metrics as metrics;
        self.r2_value = metrics.r2_score( \
            y_true = y_ref, \
            y_pred = y_pred, \
            sample_weight = None, \
            multioutput = 'uniform_average');
        self.logger.info('Accuracy Regression Score (R^2 - coefficient of determination): ' + \
            str(self.r2_value));
        return self.r2_value;
    
    def accuracy_score(self, y_ref, y_pred):
        """
        Accuracy classification score.
        In multilabel classification, this function computes subset accuracy: the set 
        of labels predicted for a sample must exactly match the corresponding set of 
        labels in y_true.        
        """
        import sklearn.metrics as metrics;
        self.accuracy_score_value = metrics.accuracy_score( \
            y_true = y_ref, 
            y_pred = y_pred, 
            normalize = True,
            sample_weight=None);
        self.logger.info('Accuracy Classification Score (R^2 - coefficient of determination): ' \
            + str(self.accuracy_score_value));
        return self.accuracy_score_value;


    def get_loss(self, type, y_ref, y_pred):
        """
        Compute the differences between the predicted score and actual score
        """
        self.logger = logging.getLogger('Model-Accuracy');    
        if type == 'regression':
            self.r2_value = self.r2_score(y_ref, y_pred);
            self.total_loss_value = self.total_loss(y_ref, y_pred);
            self.evs_value = self.explained_variance_score(y_ref, y_pred);
            self.mae_value = self.mean_absolute_error(y_ref, y_pred);
            self.mse_value = self.mean_squared_error(y_ref, y_pred);
            self.msle_value = self.mean_squared_log_error(y_ref, y_pred);
            self.mnse_value = self.median_absolute_error(y_ref, y_pred);
        else:
            self.accuracy_score_value = self.accuracy_score(y_ref, y_pred);
        return "";


    def predict(self):
        """
        Perform regression or classification of the test data
        """
        self.logger = logging.getLogger('Model-Prediction');
        self.logger.info('Input: ' + str(self.x_test));
        self.predict = self.clf.predict(self.x_test);
        self.logger.info('Prediction: ' + str(self.predict));
        self.get_loss(type = self.type, y_ref = self.y_test, y_pred = self.predict);
        self.save_data( \
            x = self.x_test, \
            y = self.predict, \
            data_file = os.path.join(self.output_dir, 'prediction_result.npz'));
        return self.predict;
        

    def load_data(self, data_file):
        """
        """
        data = np.load(data_file);
        z = data['arr_0'];
        y = z[:,0];
        x = z[:,1:];
        return x,y;

    
    def save_data(self, x, y, data_file):
        """
        """
        z = np.column_stack((y,x));
        np.savez_compressed(data_file, z)
        return data_file;


    def save_model(self, model_file):
        """
        Save obj using joblib class
        """
        self.logger.info('Saving model');
        from sklearn.externals import joblib;
        joblib.dump(self.clf, model_file);
        return model_file;


    def load_model(self):
        """
        Load obj setting using joblib class
        """
        self.logger.info('Loading model');
        from sklearn.externals import joblib;
        joblib.dump(self.clf, model_file); 
        return self.clf;


    def plot(self, mode, model, output_dir):
        """
        Plot all the different analysis
        """
        if mode == 'train':
            x_ref = self.x_train;
            y_ref = self.y_train;
            y_pred = self.clf.predict(self.x_train);
        else:
            x_ref = self.x_test;
            y_ref = self.y_test;
            y_pred = self.clf.predict(self.x_test);
        self.cross_validation_plot(mode, y_ref, y_pred, output_dir);
        return "";


    def cross_validation_plot(self, mode, y_ref, y_pred, output_dir):
        """
        Plot the different between predicted and measured data
        """
        fig, ax = plt.subplots();
        ax.scatter(y_ref, y_pred, edgecolors=(0, 0, 0));
        ax.plot([y_ref.min(), y_ref.max()], [y_pred.min(), y_pred.max()], 'k--', lw=4);
        ax.set_xlabel(mode);
        ax.set_ylabel('Predicted');
        if mode == 'train':
            ax.set_title('Cross Validation Scatter Plot for Training data');
            filename = os.path.join(output_dir, 'scatter_plot_training.pdf');
        else:
            ax.set_title('Cross Validation Scatter Plot for Test data');
            filename = os.path.join(output_dir, 'scatter_plot_testing.pdf');
        pdf = PdfPages(filename);
        pdf.savefig();
        pdf.close();
        
        return filename;


def print_help():
    """
    Print document
    """
    print(__doc__);
    exit();
    return;


if __name__ == '__main__':
    main();

