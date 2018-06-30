import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
import classificationplot as acplt


# ====== class data ======
# Class one
x1_one = np.random.uniform(0, 90, 500)
x2_one = np.random.uniform(x1_one + 10, 100, 500)
x1_one_noise = np.random.uniform(20, 80, 50)
x2_one_noise = 5/4 * (x1_one_noise - 10) + 0.1
# Class zero
x1_zero = np.random.uniform(10, 100, 500)
x2_zero = np.random.uniform(0, x1_zero-10, 500)
x1_zero_noise = np.random.uniform(20, 80, 50)
x2_zero_noise = 5/4 * (x1_zero_noise - 10) - 0.1

df1_in = pd.DataFrame(np.array([np.append(x1_one, x1_one_noise), 
                                np.append(x2_one, x2_one_noise)]).T, 
                                columns=['x1', 'x2'])
df1_in['y'] = [1]*len(df1_in.index)
df0_in = pd.DataFrame(np.array([np.append(x1_zero, x1_zero_noise), 
                                np.append(x2_zero, x2_zero_noise)]).T, 
                                columns=['x1', 'x2'])
df0_in['y'] = [0]*len(df0_in.index)
df_in = pd.concat([df0_in, df1_in])
print(df_in.head())
colors = ['red','green']
print('figure of original data')
plt.scatter(df_in['x1'], df_in['x2'], c=df_in['y'], cmap=matplotlib.colors.ListedColormap(colors))
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()

# ====== scale data ======
df_scaled = df_in[['x1','x2']].apply(lambda x: (x-x.mean())/x.std())
df_scaled['y'] = df_in['y']
X_train = df_scaled[['x1','x2']].values
y_train = df_scaled['y'].values

C = 1
r = 0.01
# ====== linear SVM ======
print('linear support vector machine')
clf = LinearSVC(C=C)
X_train = df_scaled[['x1','x2']].values
y_train = df_scaled['y'].values
clf.fit(X_train, y_train)
fig1 = acplt.ClassificationPlot('x1-x2: linear classfication', x1lab_='x1', x2lab_='x2')
fig1.plot_data_p2(X_train, y_train, clf)
plt.show()

# ====== non linear SVM ======
print('nonlinear support vector machine')
clf = SVC(C=C, gamma=r)
clf.fit(X_train, y_train)
fig1 = acplt.ClassificationPlot('x1-x2: linear classfication', x1lab_='x1', x2lab_='x2')
fig1.plot_data_p2(X_train, y_train, clf)
plt.show()