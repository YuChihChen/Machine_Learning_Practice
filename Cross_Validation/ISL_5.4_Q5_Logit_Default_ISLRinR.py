import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
import classificationplot as vplt

# get data
df_in = pd.read_csv('data/Default_ISLRinR.csv', header=0)
df_in.drop(columns='Unnamed: 0', inplace=True)
df_in['default_fac'] = pd.factorize(df_in.iloc[:, 0])[0]
print(df_in.head(5))
print(df_in.columns)

# scale the data
df_scaled = df_in[['income', 'balance']].copy()
df_scaled = df_scaled.apply(lambda x: (x-x.mean())/x.std())
df_scaled['default'] = df_in['default_fac']
print(df_scaled.head(5))
"""
Q1: Fit a logistic regression model that uses income and balance to predict default.
"""
# X_train and y_train
df_train = df_scaled.copy()
X_train = df_train.iloc[:, 0:2].values
y_train = df_train.iloc[:,  -1].values
# logistic regression 
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)
print("==================== whole data ===========================")
y_pred = classifier.predict(X_train)
print(confusion_matrix(y_pred, y_train))
print(classification_report(y_train, y_pred, digits=3))
print("false not-default of randomly guess : {}".format(sum(y_train==1)/(len(y_train))))
print("false not-default of logistic model : {}".format(sum((y_train==1) & (y_train!=y_pred))/(sum(y_pred==0))))
# scatter plot
fig1 = vplt.ClassificationPlot('Default Prediction', x1lab_='income', x2lab_='balance')
fig1.plot_data_p2(X_train, y_train, classifier)
plt.show()

"""
Q2: Using the validation set approach, estimate the test error of this model.
"""
# split into half training data and half test data
df_train = df_scaled.sample(frac=0.5, replace=False)
df_test  = df_scaled[~df_scaled.index.isin(df_train.index)]
X_train = df_train.iloc[:, 0:2].values
y_train = df_train.iloc[:,  -1].values
X_test = df_test.iloc[:, 0:2].values
y_test = df_test.iloc[:,  -1].values
# logistic regression 
classifier = LogisticRegression()
classifier.fit(X_train, y_train)
print("==================== training data ===========================")
y_pred = classifier.predict(X_train)
print(confusion_matrix(y_pred, y_train))
print(classification_report(y_train, y_pred, digits=3))
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)
print("false not-default of randomly guess : {}".format(sum(y_train==1)/(len(y_train))))
print("false not-default of logistic model : {}".format(sum((y_train==1) & (y_train!=y_pred))/(sum(y_pred==0))))
print("==================== test data ===========================")
y_pred = classifier.predict(X_test)
print(confusion_matrix(y_pred, y_test))
print(classification_report(y_test, y_pred, digits=3))
print("false not-default of randomly guess : {}".format(sum(y_test==1)/(len(y_test))))
print("false not-default of logistic model : {}".format(sum((y_test==1) & (y_test!=y_pred))/(sum(y_pred==0))))

"""
Q3: Repeat the process in (b) three times, using three different splits of the 
    observations into a training set and a validation set. Comment on the results obtained.
A3: We find that the test error has larger flutuation. 
    This is because training data has lower flutuation in biase. 
    Moreover, traing part will underestimate the noise error. 
"""
def Validation_set(df_):
    df_train = df_.sample(frac=0.5, replace=False)
    df_test  = df_[~df_.index.isin(df_train.index)]
    X_train = df_train.iloc[:, 0:2].values
    y_train = df_train.iloc[:,  -1].values
    X_test = df_test.iloc[:, 0:2].values
    y_test = df_test.iloc[:,  -1].values
    # logistic regression 
    classifier = LogisticRegression()
    classifier.fit(X_train, y_train)
    # training false non-default
    y_pred = classifier.predict(X_train)
    train_error = sum((y_train==1) & (y_train!=y_pred))/sum(y_pred==0)
    # test false non-default
    y_pred = classifier.predict(X_test)
    test_error = sum((y_test==1) & (y_test!=y_pred))/sum(y_pred==0)
    return train_error, test_error

errors_train = list()
errors_test = list()
for _ in range(10):
    err_train, err_test = Validation_set(df_scaled)
    errors_train.append(err_train)
    errors_test.append(err_test)
plt.plot(list(range(10)), errors_train, color='b', label='train')
plt.plot(list(range(10)), errors_test , color='r', label='test')
plt.legend()
plt.show()


    