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



# ================ II. Multilabel Classification ==================
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.model_selection import cross_val_predict
#from sklearn.metrics import f1_score
#y_train_large = (y_train >= 7)
#y_train_odd = (y_train % 2 == 1)
#y_multilabel = np.c_[y_train_large, y_train_odd]
#knn_clf = KNeighborsClassifier()
#knn_clf.fit(X_train, y_multilabel)
##y_train_knn_pred = cross_val_predict(knn_clf, X_train, y_train, cv=3)
#print('GOGO')
#f1_knn = f1_score(y_train, y_train_knn_pred, average="macro")
#print(f1_knn)



# ================ III. Multioutput Classification ==================
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
    
from sklearn.neighbors import KNeighborsClassifier
noise_train = np.random.randint(0, 100, (len(X_train), 784))
noise_test = np.random.randint(0, 100, (len(X_test), 784))
X_train_mod = X_train + noise_train
X_test_mod = X_test + noise_test
y_train_mod = X_train
y_test_mod = X_test
knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train_mod, y_train_mod)
noise_digit = X_test_mod[3600]
clean_digit = knn_clf.predict([noise_digit])

plt.subplot(121); plot_digits([noise_digit])
plt.subplot(122); plot_digits([clean_digit])
plt.show()
plot_digits([y_test_mod[3600]])
plt.show()
