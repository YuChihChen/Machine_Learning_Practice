import pandas as pd
import numpy as np
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
y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)

# ------ 1. Stochastic Gradient Descent (SGD) classifier ------
from sklearn.linear_model import SGDClassifier 
sgd_clf = SGDClassifier(random_state=13, max_iter=1000, tol=1e-3)
sgd_clf.fit(X_train, y_train_5)
y_pred_5 = sgd_clf.predict(X_train)
print('y_train[3600]={} and SGD say y_pred[3600]==5 is {}'
      .format(y_train[3600], sgd_clf.predict([some_digit_image.reshape(28*28)])))



# ====================== III. Performance Measures =========================

# ------ 1. Cross-Validation for testing error estimation ------
from sklearn.model_selection import cross_val_score
errors_sgd = cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy")
print('test erros of SGD from CV is ', errors_sgd)

# ------ 2. Confusion Matrix ------
from sklearn.metrics import confusion_matrix
cm_SGD = confusion_matrix(y_train_5, y_pred_5)
cm_SGD_df = pd.DataFrame(cm_SGD, index=['response_0', 'response_1'], 
                         columns=['prediction_0', 'prediction_1'])
print('=== the confusion matirx of SGD ===')
print(cm_SGD_df)

# ------ 3. PR and ROC curve ------
from sklearn.model_selection import cross_val_predict
y_measure = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3,
                              method="decision_function")
# a. PR curve
from sklearn.metrics import precision_recall_curve
precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_measure)
plt.figure(figsize=(10,8))
plt.plot(recalls, precisions, color='r')
plt.axis([0, 1, 0, 1])
plt.xlabel('recall')
plt.ylabel('precision')
plt.show()
# b. ROC curve
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
fpr, tpr, thresholds = roc_curve(y_train_5, y_measure)
auc_sgd = roc_auc_score(y_train_5, y_measure)
plt.figure(figsize=(10,8))
plt.plot(fpr, tpr, color='r', label='auc={}'.format(auc_sgd))
plt.plot([0,1], [0,1], 'k--', label='auc={}'.format(0.5))
plt.axis([0, 1, 0, 1])
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.legend()
plt.show()


