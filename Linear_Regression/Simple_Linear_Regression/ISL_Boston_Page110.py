import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

def plot_xy(x_, y_, y_pred_, title_='title', xlabel_='x', ylabel_='y' ):
    plt.scatter(x_, y_, color='red')
    plt.scatter(x_, y_pred_, color='blue')
    plt.title(title_)
    plt.xlabel(xlabel_)
    plt.ylabel(ylabel_)
    plt.show(block=False)


# get data
df_in = pd.read_csv('Boston_MASSinR.csv', header=0)
df_in.drop(columns='Unnamed: 0', inplace =True)
print(df_in.head(5))
print(df_in.columns)

X = df_in.iloc[:, 0:-1]
y = df_in.iloc[:, -1:].values


# linear fit
x = X.iloc[:, -1:].values
model = LinearRegression()
model.fit(x, y)
y_pred = model.predict(x)
print(model.intercept_, model.coef_)
# plot
plot_xy(x, y, y_pred, title_='medv vs. lstat', xlabel_='lstat', ylabel_='medv')

# linear fit2
x2 = sm.add_constant(x)
model2 = sm.OLS(y, x2)
est2 = model2.fit()
print(est2.summary())


print("The end of the code")