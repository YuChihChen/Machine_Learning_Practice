import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf


# === get data ===
df_in = pd.read_csv('data/Boston_MASSinR.csv', header=0)
df_in.drop(columns='Unnamed: 0', inplace=True)
print(df_in.shape)
print(df_in.head(5))

"""
(a) Based on this data set, provide an estimate for the population mean of medv. 
    Call this estimate \hat μ.
"""
medv_se = df_in['medv']
hat_mu = medv_se.mean()
print('estimated \hat mu is {}'.format(hat_mu))

"""
(b) Provide an estimate of the standard error of \hat μ. Interpret this result.
    Var(\hat mu) = Var(mu)/n
"""
n = len(medv_se.index)
hat_mu_std = medv_se.std()/(n**0.5)
print('estimated standard error of \hat mu is {}'.format(hat_mu_std))

"""
(c)
Question: 
    Now estimate the standard error of μˆ using the bootstrap. 
    How does this compare to your answer from (b)?
My thought before calculation:
    1.  The accuration of the estimation in method (b) depends on the accuration
        of the estimation of Var(mu). The accuration is about O(N)
    2.  For bootstrap, the varince of \hat mu is estimate by the variation of 
        these n data. Therefore, I think the The accuration is also about O(N)
    3.  Even though their arccuration are similar, the power of bootstrap is that
        we can estimate the distribution of \hat mu
"""
def bootstrap(bn_, data_se_, fun_, *args, **kwargs):
    results = list()
    for _ in range(bn_):
        data = data_se_.sample(n, replace=True)
        results.append(fun_(data, *args, **kwargs))
    return results

hat_mu_list = bootstrap(100, medv_se, lambda x: x.mean())
plt.hist(hat_mu_list)
plt.axvline(x=hat_mu, color='r', linestyle='--')
plt.plot()
plt.show()
print('estimated standard error of \hat mu from bootstrap is {}'.format(np.std(hat_mu_list)))

"""
(d) Based on your bootstrap estimate from (c), provide a 95 % confidence interval 
    for the mean of medv.
"""   
def confidence_interval(data_se_, alpha0_):
    idx = int((alpha0_/2)*n)
    data_sorted = sorted(data_se_)
    return data_sorted[idx-1], data_sorted[-idx]

mu_min, mu_max = confidence_interval(hat_mu_list, 0.05)
print('Confidence interval of \hat mu from bootstrap is [{}, {}]'.format(mu_min, mu_max))
print('Confidence interval of \hat mu from statistic is [{}, {}]'
      .format(hat_mu-2*hat_mu_std, hat_mu+2*hat_mu_std))



