import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from keras.datasets import mnist

# ============================= I. EDA =============================

# ------ 1. download and prepare data ------
(X_train_image, y_train), (X_test_image, y_test) = mnist.load_data()
print('shape of training data:', X_train_image.shape, y_train.shape)
print('shape of test     data:', X_test_image.shape, y_test.shape)
shuffle_index = np.random.permutation(60000)
X_train_image, y_train = X_train_image[shuffle_index], y_train[shuffle_index]
X_train = X_train_image.reshape(60000, 28*28)
X_test = X_test_image.reshape(10000, 28*28)

# ------ 2. plot one figure to see ------
some_digit_image = X_train_image[3600]
plt.imshow(some_digit_image, cmap = plt.cm.gray, interpolation="nearest")
plt.axis("off")
plt.show()
print('y train is : {}'.format(y_train[3600]))



# ============================= II. Training =============================
""" Notes from the book, page 94
    Some algorithms (such as Support Vector Machine classifiers) scale poorly with 
the size of the training set, so for these algorithms OvO is preferred since it 
is faster to train many classifiers on small training sets than training few 
classifiers on large training sets. For most binary classification algorithms, 
however, OvA is preferred.
    Scikit-Learn detects when you try to use a binary classification algorithm 
for a multiclass classification task, and it automatically runs OvA (except for 
SVM classifiers for which it uses OvO). 
    For classifier that can fit multiple class automatically, such as random forest, 
    logistic regression ... ect, Scikit-Learn did not have to run OvA or OvO because 
    the classifiers can directly classify instances into multiple classes.
"""
# ------ 1. Stochastic Gradient Descent (SGD) classifier ------
# a. OvA classifier
from sklearn.linear_model import SGDClassifier 
sgd_clf = SGDClassifier(random_state=13, max_iter=1000, tol=1e-3)
sgd_clf.fit(X_train, y_train)
y_pred = sgd_clf.predict(X_train)
print('y_train[3600]={} and SGD_ova say y_pred[3600]={}'
      .format(y_train[3600], sgd_clf.predict([some_digit_image.reshape(28*28)])))
# b. OvO classifier
from sklearn.multiclass import OneVsOneClassifier
sgd_clf_ovo = OneVsOneClassifier(SGDClassifier(random_state=13, max_iter=1000, tol=1e-3))
sgd_clf_ovo.fit(X_train, y_train)
y_pred_ovo = sgd_clf_ovo.predict(X_train)
print('y_train[3600]={} and SGD_ovo say y_pred[3600]={}'
      .format(y_train[3600], sgd_clf_ovo.predict([some_digit_image.reshape(28*28)])))



# ====================== III. Performance Measures =========================

# ------ 1. Cross-Validation for testing error estimation ------
from sklearn.model_selection import cross_val_score
errors_sgd = cross_val_score(sgd_clf, X_train, y_train, cv=3, scoring="accuracy")
print('test erros of SGD from CV is ', errors_sgd)

# ------ 2. Confusion Matrix ------
from sklearn.metrics import confusion_matrix
cm_sgd = confusion_matrix(y_train, y_pred)
cm_sgd_ovo = confusion_matrix(y_train, y_pred)
print('====== confusion matrix of model SGD with OvA ======')
print(cm_sgd)
plt.matshow(cm_sgd, cmap=plt.cm.gray)
plt.show()
print('====== confusion matrix of model SGD with OvO ======')
print(cm_sgd_ovo)
plt.matshow(cm_sgd_ovo, cmap=plt.cm.gray)
plt.show()

print('====== errors of model SGD ======')
"""
    Now you can clearly see the kinds of errors the classifier makes. 
Remember that rows represent actual classes, while columns represent predicted 
classes. The columns for classes 8 and 9 are quite bright, which tells you that 
many images get misclassified as 8s or 9s. Similarly, the rows for classes 8 and 
9 are also quite bright, telling you that 8s and 9s are often confused with other 
digits. Conversely, some rows are pretty dark, such as row 1: this means that most 
1s are classified correctly (a few are confused with 8s, but that’s about it). 
Notice that the errors are not perfectly symmetrical; for example, there are more 
5s misclassified as 8s than the reverse.
    Analyzing the confusion matrix can often give you insights on ways to improve 
your classifier. Looking at this plot, it seems that your efforts should be spent 
on improving classification of 8s and 9s, as well as fixing the specific 3/5 confusion.
"""
row_sums = cm_sgd.sum(axis=1, keepdims=True)
norm_conf_sgd = cm_sgd / row_sums
np.fill_diagonal(norm_conf_sgd, 0)
plt.matshow(norm_conf_sgd, cmap=plt.cm.gray)
plt.show()

# --- 3. 3/5 confusion ---
def plot_digits(instances, images_per_row=10, **options):
    size = 28
    images_per_row = min(len(instances), images_per_row)
    images = [instance.reshape(size,size) for instance in instances] #转换成100个像素阵
    n_rows = (len(instances) - 1) // images_per_row + 1
    row_images = []
    n_empty = n_rows * images_per_row - len(instances)
    images.append(np.zeros((size, size * n_empty)))
    for row in range(n_rows):
        rimages = images[row * images_per_row : (row + 1) * images_per_row]
        row_images.append(np.concatenate(rimages, axis=1))      #实现list的reshape
    image = np.concatenate(row_images, axis=0)  
    plt.imshow(image, cmap = matplotlib.cm.binary, **options)
    plt.axis("off")
    
cl_a, cl_b = 3, 5
X_aa = X_train[(y_train == cl_a) & (y_pred == cl_a)]
X_ab = X_train[(y_train == cl_a) & (y_pred == cl_b)]
X_ba = X_train[(y_train == cl_b) & (y_pred == cl_a)]
X_bb = X_train[(y_train == cl_b) & (y_pred == cl_b)]
plt.figure(figsize=(8,8))
plt.subplot(221); plot_digits(X_aa[:25], images_per_row=5)
plt.subplot(222); plot_digits(X_ab[:25], images_per_row=5)
plt.subplot(223); plot_digits(X_ba[:25], images_per_row=5)
plt.subplot(224); plot_digits(X_bb[:25], images_per_row=5)
plt.show()