# Supervised learning
* [Generalized Linear Models](#generalized-linear-models)
* Linear and Quadratic Discriminant Analysis
* Kernel ridge regression
1.4. Support Vector Machines
* [Stochastic Gradient Descent](#stochastic-gradient-descent)
* [Nearest Neighbors](#nearest-neighbors)
* Gaussian Processes
1.8. Cross decomposition
1.9. Naive Bayes
1.10. Decision Trees
1.11. Ensemble methods
1.12. Multiclass and multilabel algorithms
1.13. Feature selection
1.14. Semi-Supervised
1.15. Isotonic regression
1.16. Probability calibration
1.17. Neural network models (supervised)
# Unsupervised learning
2.1. Gaussian mixture models
2.2. Manifold learning
2.3. Clustering
2.4. Biclustering
2.5. Decomposing signals in components (matrix factorization problems)
2.6. Covariance estimation
2.7. Novelty and Outlier Detection
2.8. Density Estimation
2.9. Neural network models (unsupervised)


## Generalized Linear Models 

## Stochastic Gradient Descent
* Stochastic Gradient Descent (SGD) is a simple yet very efficient approach to discriminative learning of linear classifiers under convex loss functions such as (linear) Support Vector Machines and Logistic Regression. 
* Even though SGD has been around in the machine learning community for a long time, it has received a considerable amount of attention just recently in the context of large-scale learning.
* SGD has been successfully applied to large-scale and sparse machine learning problems often encountered in text classification and natural language processing. Given that the data is sparse, the classifiers in this module easily scale to problems with more than 10^5 training examples and more than 10^5 features.
### The advantages of Stochastic Gradient Descent are:
* Efficiency.
* Ease of implementation (lots of opportunities for code tuning).
### The disadvantages of Stochastic Gradient Descent include:
* SGD requires a number of hyperparameters such as the regularization parameter and the number of iterations.
* SGD is sensitive to feature scaling.

## Nearest Neighbors
sklearn.neighbors provides functionality for unsupervised and supervised neighbors-based learning methods. Unsupervised nearest neighbors is the foundation of many other learning methods, notably manifold learning and spectral clustering. Supervised neighbors-based learning comes in two flavors: classification for data with discrete labels, and regression for data with continuous labels.
The principle behind nearest neighbor methods is to find a predefined number of training samples closest in distance to the new point, and predict the label from these. The number of samples can be a user-defined constant (k-nearest neighbor learning), or vary based on the local density of points (radius-based neighbor learning). The distance can, in general, be any metric measure: standard Euclidean distance is the most common choice. Neighbors-based methods are known as non-generalizing machine learning methods, since they simply “remember” all of its training data (possibly transformed into a fast indexing structure such as a Ball Tree or KD Tree.).
Despite its simplicity, nearest neighbors has been successful in a large number of classification and regression problems, including handwritten digits or satellite image scenes. Being a non-parametric method, it is often successful in classification situations where the decision boundary is very irregular.
The classes in sklearn.neighbors can handle either Numpy arrays or scipy.sparse matrices as input. For dense matrices, a large number of possible distance metrics are supported. For sparse matrices, arbitrary Minkowski metrics are supported for searches.


## Gaussian Processes
* Gaussian Processes (GP) are a generic supervised learning method designed to solve regression and probabilistic classification problems.
#### The advantages of Gaussian processes are:
* The prediction interpolates the observations (at least for regular kernels).
* The prediction is probabilistic (Gaussian) so that one can compute empirical confidence intervals and decide based on those if one should refit (online fitting, adaptive fitting) the prediction in some region of interest.
* Versatile: different kernels can be specified. Common kernels are provided, but it is also possible to specify custom kernels.
#### The disadvantages of Gaussian processes include:
* They are not sparse, i.e., they use the whole samples/features information to perform the prediction.
* They lose efficiency in high dimensional spaces – namely when the number of features exceeds a few dozens.


## sklearn.svm: Support vector machines (SVMs) 
* a set of supervised learning methods used for classification, regression and outliers detection.

#### The advantages of support vector machines are:
* Effective in high dimensional spaces.
* Still effective in cases where number of dimensions is greater than the number of samples.
* Uses a subset of training points in the decision function (called support vectors), so it is also memory efficient.
* Versatile: different Kernel functions can be specified for the decision function. Common kernels are provided, but it is also possible to specify custom kernels.
### The disadvantages of support vector machines include:
* If the number of features is much greater than the number of samples, avoid over-fitting in choosing Kernel functions and regularization term is crucial.
* SVMs do not directly provide probability estimates, these are calculated using an expensive five-fold cross-validation (see Scores and probabilities, below).

### Estimators
#### a. svm.LinearSVC() - Linear Support Vector Classification.
#### b. svm.LinearSVR([epsilon, tol, C, loss, …])	Linear Support Vector Regression.
#### c. svm.NuSVC() - Nu-Support Vector Classification.
* decision_function(X) - Distance of the samples X to the separating hyperplane.
* fit(X, y[, sample_weight]) - Fit the SVM model according to the given training data.
* get_params([deep]) - Get parameters for this estimator.
* predict(X) - Perform classification on samples in X.
* score(X, y[, sample_weight]) - Returns the mean accuracy on the given test data and labels.
* set_params(**params) - Set the parameters of this estimator.
#### d. svm.NuSVR() - Nu Support Vector Regression.
* fit(X, y[, sample_weight]) - Fit the SVM model according to the given training data.
* get_params([deep]) - Get parameters for this estimator.
* predict(X) - Perform regression on samples in X.
* score(X, y[, sample_weight]) - Returns the coefficient of determination R^2 of the prediction.
* set_params(**params) - Set the parameters of this estimator.
#### e. svm.SVR([kernel, degree, gamma, coef0, tol, …])	Epsilon-Support Vector Regression.



#### f. svm.OneClassSVM([kernel, degree, gamma, …]) - Unsupervised Outlier Detection.
* decision_function(X) - Distance of the samples X to the separating hyperplane.
* fit(X, y[, sample_weight]) - Fit the SVM model according to the given training data.
* get_params([deep]) - Get parameters for this estimator.
* predict(X) - Perform classification on samples in X.
* set_params(**params) - Set the parameters of this estimator.

#### svm.SVC([C, kernel, degree, gamma, coef0, …])	C-Support Vector Classification.

* svm.l1_min_c(X, y[, loss, fit_intercept, …])	Return the lowest bound for C such that for C in (l1_min_C, infinity) the model is guaranteed not to be empty.

#### Low-level methods
* svm.libsvm.cross_validation	Binding of the cross-validation routine (low-level routine)
* svm.libsvm.decision_function	Predict margin (libsvm name for this is predict_values)
* svm.libsvm.fit	Train the model using libsvm (low-level method)
* svm.libsvm.predict	Predict target values of X given a model (low-level method)
* svm.libsvm.predict_proba	Predict probabilities

#### Examples
````
# fit the model
clf = svm.NuSVC()
clf.fit(X, Y)
# plot the decision function for each datapoint on the grid
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
````


## skimage.color.adapt_rgb
**adapt_rgb**

**each_channel**
- Pass each of the RGB channels to the filter one-by-one, and stitch the results back into an RGB image.


 hsv_value
