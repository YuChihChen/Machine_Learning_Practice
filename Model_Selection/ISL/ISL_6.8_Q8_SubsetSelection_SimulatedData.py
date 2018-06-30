import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

def forward_selected(data, response):
    """Linear model designed by forward selection.

    Parameters:
    -----------
    data : pandas DataFrame with all possible predictors and response

    response: string, name of response column in data

    Returns:
    --------
    model: an "optimal" fitted statsmodels linear model
           with an intercept
           selected by forward selection
           evaluated by adjusted R-squared
    """
    remaining = set(data.columns)
    remaining.remove(response)
    selected = []
    current_score, best_new_score = 0.0, 0.0
    while remaining and current_score == best_new_score:
        scores_with_candidates = []
        for candidate in remaining:
            formula = "{} ~ {} + 1".format(response,
                                           ' + '.join(selected + [candidate]))
            score = smf.ols(formula, data).fit().rsquared_adj
            scores_with_candidates.append((score, candidate))
        scores_with_candidates.sort()
        best_new_score, best_candidate = scores_with_candidates.pop()
        if current_score < best_new_score:
            remaining.remove(best_candidate)
            selected.append(best_candidate)
            current_score = best_new_score
    formula = "{} ~ {} + 1".format(response,
                                   ' + '.join(selected))
    model = smf.ols(formula, data).fit()
    print('Best model from forward selection is: {}'.format(formula))
    return model


def shrinkage_CV(df_in_, response_, fit_formula_, lambda_, L1_wt_=1.0, K_=5 ):
    df_scaled = df_in_.drop([response_],1)
    df_scaled = df_scaled.apply(lambda x: (x-x.mean())/x.std(), axis=1)
    df_scaled[response_] = df_in_[response_]
    df_shffle = df_scaled.sample(frac=1, replace=False)
    bin_size = len(df_scaled.index)//K_
    label = list()
    for k in range(K_-1):
        label.extend([k]*bin_size)
    label.extend([K_-1]*(len(df_scaled.index)-len(label)))
    df_shffle['label'] = label
    error_list = list()
    for k in range(K_):
        df_test  = df_shffle[df_shffle.label==k]
        df_train = df_shffle[~df_shffle.index.isin(df_test.index)]
        model = smf.ols(fit_formula_, df_train).fit_regularized(alpha=lambda_, L1_wt=L1_wt_)
        y_pre = model.predict(df_test.drop([response_,'label'],1))
        error =  ((y_pre-df_test[response_])**2)
        error_list.extend(error)
    return np.mean(error_list)


def shrinkage_method(data, response, lamda_list, L1_wt=1.0):
    # 1. get formula
    remaining = set(data.columns)
    remaining.remove(response)
    formula = 'y ~ ' + ' + '.join([x for x in remaining])
    # 2. cross validation
    error_list = list()
    for ld in lamda_list:
        err = shrinkage_CV(data, response, formula, ld, L1_wt_=L1_wt, K_=5 )
        error_list.append(err)
    # 3. plot the diagram
    plt.scatter(lambda_list, error_list)
    plt.show()
    # 4. find the best
    err_opt = error_list[0]
    lda_opt = lamda_list[0]
    for i in range(1, len(lamda_list)):
        if error_list[i] < err_opt:
            err_opt = error_list[i]
            lda_opt = lamda_list[i]
    model = smf.ols(formula, data).fit_regularized(alpha=lda_opt, L1_wt=L1_wt)
    return model

"""
In this exercise, we will generate simulated data, 
and will then use this data to perform best subset selection.
"""

"""
(a) Use the rnorm() function to generate a predictor X of length n = 100, 
    as well as a noise vector ε of length n = 100.
"""
n = 500
mean = 0
variance = 1
x = np.random.normal(loc=mean, scale=np.sqrt(variance), size=n)
e = np.random.normal(loc=mean, scale=np.sqrt(variance), size=n)

"""
(b) Generate a response vector Y of length n = 100 according to the model
    Y = β0 + β1*X + β2*X^2 + β3*X^3 + ε,
    where β0, β1, β2, and β3 are constants of your choice.
"""
#y = 3 + 1*x + 2*(x**2) + 3*(x**3) + e
y = 3 + 3*(x**4) + e
plt.scatter(x,y)
plt.show()


"""
(c) Use the regsubsets() function to perform best subset selection in order to 
    choose the best model containing the predictors X, X^2, . . . , X^10. 
    What is the best model obtained according to Cp, BIC, and adjusted R2? 
    Show some plots to provide evidence for your answer, and report the 
    coefficients of the best model ob- tained. Note you will need to use 
    the data.frame() function to create a single data set containing both 
    X and Y .
"""
df = pd.DataFrame(y, columns=['y'])
for i in range(1,11,1):
    df['x'+str(i)] = x**i
    #df['x'+str(i)] =np.random.normal(loc=i, scale=np.sqrt(i*variance), size=n)
forward_model = forward_selected(df, 'y') 


"""
(e) Now fit a lasso model to the simulated data, again using X,X2, . . . , X 10 
    as predictors. Use cross-validation to select the optimal value of λ. 
    Create plots of the cross-validation error as a function of λ. 
    Report the resulting coefficient estimates, and discuss the results obtained.
"""
lambda_list = np.arange(0, 1, 0.1)
best_lasso = shrinkage_method(df, 'y', lambda_list)
print(best_lasso.params)


