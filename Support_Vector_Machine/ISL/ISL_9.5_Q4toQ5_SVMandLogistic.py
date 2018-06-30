import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import crossvalidation as acv
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
import classificationplot as acplt



# ====== get data ======
x1 = np.random.uniform(0, 1, 500)-0.5
x2 = np.random.uniform(0, 1, 500)-0.5
df_in = pd.DataFrame(np.array([x1,x2]).T, columns=['x1', 'x2'])
df_in['y'] = ((x1**2 - x2**2)>0).astype(int)
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

# ====== linear logistic regression ======
print('linear logistic regression')
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)
fig1 = acplt.ClassificationPlot('x1-x2: linear classfication', x1lab_='x1', x2lab_='x2')
fig1.plot_data_p2(X_train, y_train, classifier)
plt.show()

# ===== non-linear logistic regression ======
print('non-linear logistic regression')
df_scaled['x1_2'] = df_scaled['x1']**2
df_scaled['x2_2'] = df_scaled['x2']**2
df_scaled['x1x2'] = df_scaled['x1']*df_scaled['x2']
cols = list(df_scaled.columns)
cols.remove('y')
X_train = df_scaled[cols].values
y_train = df_scaled['y'].values
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_train)
plt.scatter(df_scaled['x1'], df_scaled['x2'], c=y_pred, cmap=matplotlib.colors.ListedColormap(colors))
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()

# ====== linear SVM ======
print('linear support vector machine')
clf = LinearSVC(random_state=0)
X_train = df_scaled[['x1','x2']].values
y_train = df_scaled['y'].values
clf.fit(X_train, y_train)
fig1 = acplt.ClassificationPlot('x1-x2: linear classfication', x1lab_='x1', x2lab_='x2')
fig1.plot_data_p2(X_train, y_train, clf)
plt.show()

# ====== non linear SVM ======
print('nonlinear support vector machine')
clf = SVC()
clf.fit(X_train, y_train)
fig1 = acplt.ClassificationPlot('x1-x2: linear classfication', x1lab_='x1', x2lab_='x2')
fig1.plot_data_p2(X_train, y_train, clf)
plt.show()