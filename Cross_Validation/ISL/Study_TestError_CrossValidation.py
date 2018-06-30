import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

""" Part 1 : True test error
===============================================================================
In this part, we estimate the true test error, as well as the model variance,
model bias and noise variance 
Note 1:
    true_test_mse: the error from E_{test,train}[(y - yhat)^2]
    true_test_err: the error from Var(model)+baise+Var(noise)
    In theory, true_test_mse = true_test_err
Note 2:
    If we set true funtion to be x-2*(x**2)+(x**3), we will find the our 
    true_test_mse > true_test_error in practice. This is becuase that, we draw
    xs in true_test_mse n_*m_ times, but we draw xs in true_test_error only n_
    times. Therefore, true_test_mse has higher probability to having large |x|,
    which will contribute a lot in bias. 
    From my point of view, true_test_mse is more close to the true test error 
    because true_test_error may underestimate the bias error. 
    For making sure my point of view, we can condut true_test_error funtion m_
    times and to see what happen. 
Note 3:
    Therefore, we will use true_test_mse as out true test error, and treat the 
    true_model_var and noise_var in true_test_error as true. In this logic, we 
    can get 
    true_model_bias = true_test_mse - true_model_var - noise_var
===============================================================================
"""
# ======= funtion to generate x and y ======
def generate_xydf(n_, yfun_, noise_variance_):
    mean = 0
    variance = 1
    x = np.random.normal(loc=mean, scale=np.sqrt(variance), size=n_)
    y = yfun_(x)+np.random.normal(loc=mean, scale=np.sqrt(noise_variance_), size=n_)
    return pd.DataFrame({'x': x, 'y': y})


# ======= parameters ======
n = 100    
m = 100
noise_variance = 2
fun_ld = lambda x: x-2*(x**2)+0.5*(x**3)
model_formula = 'y ~ x + np.power(x, 2)'


# ======= fitting model with linear regression ======
df_train = generate_xydf(n, fun_ld, noise_variance)
df_test  = generate_xydf(n, fun_ld, noise_variance)
model = smf.ols(formula=model_formula, data=df_train).fit()
y_pred = model.predict(df_test.iloc[:, 0])
MSE = ((y_pred-df_test.iloc[:, 1])**2).mean()
plt.scatter(df_train.iloc[:,0], df_train.iloc[:,1], label='data')
plt.scatter(df_test.iloc[:,0], y_pred, color='r', label='prediciton')
plt.legend()
plt.show()


# ====== funtion to get real testing error for model with n samples ====== 
def true_test_mse(n_, yfun_, noise_variance_, fit_formula_, m_): 
    m = m_
    true_SSRs = list()
    xs = list()
    for _ in range(m):
        # get a model with n samples
        df_train = generate_xydf(n_, yfun_, noise_variance_)
        model = smf.ols(formula=fit_formula_, data=df_train).fit()
        # calculate the testing MSE with given model(fixed training sample)
        df_test = generate_xydf(n_, yfun_, noise_variance_)
        y_pred = model.predict(df_test.iloc[:, 0])
        SSR = ((y_pred-df_test.iloc[:, 1])**2).sum()
        true_SSRs.append(SSR)
        xs.extend(df_test.iloc[:, 0].values)
    plt.hist(xs)
    plt.title('xs in mse')
    plt.show()
    return sum(true_SSRs)/(m_*n_)

def true_test_error(n_, yfun_, noise_variance_, fit_formula_, m_): 
    x = np.random.normal(loc=0, scale=np.sqrt(1), size=n_) # fixed x
    y_preds = list()
    for _ in range(m_):
        y_train = yfun_(x)+np.random.normal(loc=0, scale=np.sqrt(noise_variance_), size=n_)
        df_train = pd.DataFrame({'x': x, 'y': y_train})
        model = smf.ols(formula=fit_formula_, data=df_train).fit()
        y_pred = model.predict(df_train.iloc[:, 0])
        y_preds.append(y_pred)    
    # get model_var, model_biase, noise_var
    # model_var
    df_y_preds = pd.DataFrame(y_preds)
    true_model_means = df_y_preds.apply(np.mean)         # E[yhat]
    true_model_var = df_y_preds.apply(np.var).mean()     # Var(yhat)
    # model_biase
    true_model_biase = ((true_model_means - yfun_(x))**2).mean()
    # true error
    true_error = true_model_var+true_model_biase+noise_variance_
    # plot histogram
    plt.hist(x)
    plt.title('xs in error')
    plt.show()
    # plot the diagram
    plt.scatter(x, yfun_(x)        , color='b', label='true  funciton')
    plt.scatter(x, true_model_means, color='r', label='model funciton')
    plt.legend()
    plt.show()
    return true_error, true_model_var, true_model_biase,noise_variance_ 

# ====== True Test Error ======
true_mse = true_test_mse(n, fun_ld, noise_variance, model_formula, m_=m)
true_err = true_test_error(n, fun_ld, noise_variance, model_formula, m_=m) 
print('The MSE from a given model and give n samples is {}'.format(MSE))       
print("true err = {} = {} + {} + {}".format(*true_err))
print('true MSE = {}'.format(true_mse))
print("================== colusion ====================")
test_error = true_mse
model_vara = true_err[1]
model_bias = test_error-true_err[1]-true_err[3]
noise_vara = true_err[3]
print("True test error     : {}".format(test_error))
print("True model variance : {}".format(model_vara))
print("True model bias     : {}".format(model_bias))
print("True noise variance : {}".format(noise_vara))



""" Part 2: Data for Part 3 to Part 5 
===============================================================================
Here we prepare n samples for our analysis, assuming that we only have these
n samples to conduct analysis.
===============================================================================
"""
df_in = generate_xydf(n, fun_ld, noise_variance)
print(df_in.head())



""" Part 3: Using validation set to estimate test error 
===============================================================================
In this part, we estimate the test error by using validation set, that is, we 
seperate data into training and test part with equl size.
Discussion:
    Suppose we divide the data into A, B two parts, what we will do is training 
    A and testing B, and then training B and testing A. Then the average of test
    error would be our estimation of true testing error.
    1. we only cahnge our model data twice, the model_variance will not accurate,
       therefore, we will underestimate the model_variance
    2. we only use half data to train our model. Therefore, our model will have
       high bias. 
    3. For the noise part, we already use all of data to estimate. Therefore, 
       this would be O(N) accurate and will be similar with part 4 and 5.
    From 1, 2, 3, because the order of 1 is relatively small. I guess the estimated
    error will be larger than the true test error.
result:
    The result shows the estimated error is smaller than true test error. 
    This is again because that the true test error include the bias error for 
    large |x|. However, the x in part 3 to 5 do not. This can be seen from the 
    histogram plots.
===============================================================================
"""
plt.hist(df_in.iloc[:, 0])
plt.title('xs for part 3, 4, 5')
plt.show()
def validation_set(df_in_, fit_formula_):
    df_A = df_in_.sample(frac=0.5, replace=False)
    df_B = df_in_[~df_in_.index.isin(df_A.index)]
    # A train B test
    modelA = smf.ols(formula=fit_formula_, data=df_A).fit()
    prediB = modelA.predict(df_B.iloc[:, 0])
    errorB =  ((prediB-df_B.iloc[:, 1])**2).sum()
    # B train A test
    modelB = smf.ols(formula=fit_formula_, data=df_B).fit()
    prediA = modelB.predict(df_A.iloc[:, 0])
    errorA =  ((prediA-df_A.iloc[:, 1])**2).sum()
    return (errorA+errorB)/len(df_in_.index)

error_validation_set = validation_set(df_in, model_formula)
print('test error from validation set is: {}'.format(error_validation_set))


""" Part 4: Using LOOCV to estimate test error 
===============================================================================
In this part, we estimate the test error by using LOOCV
Discussion:
    1. Since every time we only change one data, the model will not change too much.
       Therefor, the model_variance will not accurate,and we will underestimate 
       the model_variance
    2. we use all data to train our model. Therefore, our model will have
       lower bias error than part 3. 
    3. For the noise part, we already use all of data to estimate. Therefore, 
       this would be O(N) accurate and will be similar with part 4 and 5.
    From 1, 2, 3, I guess the estimated error will be smaller than the part 3.
result:
    1. The result shows the estimated error is smaller than true test error. 
       This is again because that the true test error include the bias error for 
       large |x|. However, the x in part 3 to 5 do not. This can be seen from the 
       histogram plots.
===============================================================================
"""
def LOOCV(df_in_, fit_formula_):
    error_list = list()
    for i in range(len(df_in.index)):
        df_test  = df_in_[ df_in_.index.isin([i])]
        df_train = df_in_[~df_in_.index.isin([i])]    
        model = smf.ols(formula=fit_formula_, data=df_train).fit()
        y_pre = model.predict(df_test.iloc[:, 0])
        error =  ((y_pre-df_test.iloc[:, 1])**2).iloc[0]
        error_list.append(error)
    return np.mean(error_list)


error_LOOCV = LOOCV(df_in, model_formula)
print('test error from LOOCV is: {}'.format(error_LOOCV))  
    
    
""" Part 5: Using K-fold CV to estimate test error 
===============================================================================
In this part, we estimate the test error by using 5-fold CV
Discussion:
    1. The estimation of model_variance will be better than 3 and 4 
    2. we use 5/4 data to train our model. Therefore, our model will have
       lower bias error than part 3, but higher than part 4. 
    3. For the noise part, we already use all of data to estimate. Therefore, 
       this would be O(N) accurate and will be similar with part 3 and 4.
    From 1, 2, 3, I guess the estimated error will be smaller than the part 3, 
    but larger than part 4.
result:
    1. The result shows the estimated error is smaller than true test error. 
       This is again because that the true test error include the bias error for 
       large |x|. However, the x in part 3 to 5 do not. This can be seen from the 
       histogram plots.
===============================================================================
"""
def K_fold(df_in_, fit_formula_, K_):
    df_shffle = df_in_.sample(frac=1, replace=False)
    bin_size = len(df_in_.index)//K_
    label = list()
    for k in range(K_-1):
        label.extend([k]*bin_size)
    label.extend([K_-1]*(len(df_in_.index)-len(label)))
    df_shffle['label'] = label
    error_list = list()
    for k in range(K_):
        df_test  = df_shffle[df_shffle.label==k]
        df_train = df_shffle[~df_shffle.index.isin(df_test.index)]
        model = smf.ols(formula=fit_formula_, data=df_train).fit()
        y_pre = model.predict(df_test.iloc[:, 0])
        error =  ((y_pre-df_test.iloc[:, 1])**2)
        error_list.extend(error)
    return np.mean(error_list)


error_K_fold= K_fold(df_in, model_formula, K_=5)
print('test error from K_fold is: {}'.format(error_K_fold))  