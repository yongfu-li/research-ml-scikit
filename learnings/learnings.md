
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
