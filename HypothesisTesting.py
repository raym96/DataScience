import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from sklearn.metrics import mean_squared_error, r2_score


def mult_normality(data, cols, method='shapiro', hist=True, qq=True, prt=False):
    table = None
    res = []
    for col in cols:
        X = data[col]
        tst = test()
        if hist:
            hist_name = 'Distribution of '+col
        
        tst.sgl_normality(X=X, method=method, hist=hist, qq=qq, prt=prt)
        res.append(tst.res)
        
        if method == 'anderson':
            tmp_table = tst.table
            tmp_table.index = [col +"_"+ c for c in tmp_table.index]
            table = pd.concat([table,tmp_table],axis=0)
        elif method == 'shapiro':
            tmp_table = tst.table
            tmp_table.index = [col]
            table = pd.concat([table,tmp_table],axis=0)
        elif method == 'kolmogorov':
            tmp_table = tst.table
            tmp_table.index = [col]
            table = pd.concat([table,tmp_table],axis=0)
    
    tst = test()  
    tst.res = res
    tst.table = table
    
    return tst
            
        
class test:
    def __init__(self):
        self.res = None
        self.type = None
        self.table = None 
    
    # test normality of one single sample
    def sgl_normality(self, X, method='shapiro', hist_name=None, hist=True, qq=True, prt=True):
        self.type = 'normality test'

        # plot the distribution if hist==True
        if hist:
            plt.figure(figsize=(8, 6), dpi=100)
            if hist_name is not None:
                plt.title(hist_name)
            sns.distplot(X, rug=False, hist=True, kde=True, label = 'Sample distribution')
            support = np.linspace(min(X), max(X), num=100)
            dens = norm.pdf(support, loc=np.mean(X), scale=np.std(X))
            sns.lineplot(support, dens, label='Gaussian with sample mean and std')
        
        # plot the qq-plot
        if qq:
            plt.figure(figsize=(8, 6), dpi=100)
            stats.probplot(X, plot=sns.mpl.pyplot)
        
        # normality test
        if method == 'anderson': 
            # anderson test 
            # return decisions for different significance levels
            self.res = stats.anderson(X, dist='norm')
            self.table = pd.DataFrame(np.concatenate(([self.res.significance_level],
                                                 [self.res.critical_values],[self.res.critical_values]),axis=0))
            self.table.index = ['sig_level','crit_vals','decision']
            #print("ad_stat:", res.statistic)
            self.table.loc['decision'] = (self.table.loc['crit_vals']<self.res.statistic)
            self.table.loc['decision'] = self.table.loc['decision'].apply(lambda x: 'Rej' if x else 'no Rej')
        
        if method == 'shapiro':
            # shapiro test
            self.res = stats.shapiro(X)
            self.table = pd.DataFrame([['{:.4e}'.format(float(self.res[0])),'{:.4e}'.format(float(self.res[1]))]])
            self.table.columns = ['test stat','p-value']
            if prt:
                print("Shapiro test statistic:",'{:.4e}'.format(float(self.res[0])) )
                print("Shapiro test p-value:",'{:.4e}'.format(float(self.res[1])) )
            
        if method == 'kolmogorov':
            # Kolmogorov-Smirnov test
            self.res = stats.kstest(X, 'norm')
            self.table = pd.DataFrame([['{:.4e}'.format(float(self.res[0])),'{:.4e}'.format(float(self.res[1])) ]])
            self.table.columns = ['test stat','p-value']
            if prt:
                print("Kolmogorov-Smirnov test statistic:",'{:.4e}'.format(float(self.res[0])) )
                print("Kolmogorov-Smirnov test p-value:",'{:.4e}'.format(float(self.res[1])) )
                       
    # 1-sample t tests whether mean is mu, requires normality
    # but if sample size > 15, robust
    # or use Wilcoxon Test by setting normality=False
    # data is a dataframe, cols indicating which columns to test
    # mu is the target mean array
    def t_1sample(self, data, cols, mu, normality=True):
        self.type = 'one-sample t test'
        
        if normality:
            # usual t-test
            self.res = stats.ttest_1samp(data[cols], mu)
            self.table = pd.concat((pd.DataFrame(['{:.4e}'.format(float(c)) for c in self.res[0]]),
                                    pd.DataFrame(['{:.4e}'.format(float(c)) for c in self.res[1]])),axis=1)
            self.table.columns = ['test stat','p-value']
            self.table.index = cols
        else:
            # wilcoxon t-test
            self.res = []
            for i in range(len(mu)):
                x = data[cols[i]]
                tmp_res = stats.wilcoxon(x - mu[i])
                self.res.append(tmp_res)
                tmp_table = pd.DataFrame(['{:.4e}'.format(tmp_res[0]),'{:.4e}'.format(tmp_res[1])]).transpose()
                self.table = pd.concat((self.table,tmp_table),axis=0)
            self.table.columns = ['test stat','p-value']
            self.table.index = cols
            
    # test whether group 1 and 2 has different mean
    # perform welch-t if normality == False
    # plot violin plot of two samples if plot == True
    # print results if prt == True
    def t_2sample(self, gp1, gp2, normality=True, plot=True, prt=True, sig_level=0.05):
        self.type = '2-sample t test'
        self.res = stats.ttest_ind(gp1, gp2, equal_var=normality, nan_policy='omit')
        table = pd.DataFrame(['{:.4e}'.format(float(c)) for c in self.res]).transpose()
        table.columns = ['test stat', 'p-value']
        self.table = table
        reject = False
        if self.res[1] < sig_level:
            reject = True
        
        if prt:
            print("2-sample test statistic: ",'{:.4e}'.format(float(self.res[0])))
            print("2-sample test p-value: ",'{:.4e}'.format(float(self.res[1])))
            if reject:
                print("have confidence to assert the groups have different means at significant level",sig_level)
            else:
                print("have no evidence to say the groups have different means at significant level",sig_level)
        
        if plot:
            gp = pd.DataFrame({'group1':list(X.iloc[0:100]),
                   'group2':list(X.iloc[100:200])})
            gpp = gp.melt().assign(x='vars')
            plt.figure(figsize=(8, 6), dpi=100)
            plt.title('violin plot of two groups')
            sns.violinplot(data=gpp, x='x', y='value', 
                           hue='variable', split=True, inner='quart')
    
    # test H0: submodel (with p features in cols1) 
    #   vs H1: full model (q>p features in col2) in linear regression
    # calculate the residuals of the two Rsub, Rful (Rsub >= Rful)
    # under H0, (Rsub-Rful)/Rful * n-q/q-p ~ F_q-p,n-q
    def full_model_test(self, data, y, cols1, cols2, add_const=True, summary = False, prt = True, sig_lev = 0.05):
        if add_const:
            subX_ = sm.add_constant(data[cols1])
            fulX_ = sm.add_constant(data[cols2])
        n = len(y)
        p = len(cols1) + 1
        q = len(cols2) + 1
        submd = sm.OLS(y.values, np.asarray(subX_))
        fulmd = sm.OLS(y.values, np.asarray(fulX_))
        subres = submd.fit()
        fulres = fulmd.fit()
        self.res = [subres, fulres]
        
        if summary:
            print("summary of submodel:")
            print(subres.summary()) # Robusted Results
            print("summary of fullmodel:")
            print(fulres.summary())
        
        sub_resid = np.sum(subres.resid**2)
        ful_resid = np.sum(fulres.resid**2)
        F_stat = ((sub_resid-ful_resid)/(q-p)) / (ful_resid/(n-q))
        p_val = 1 - stats.f.cdf(F_stat, q-p, n-q)
        
        table = pd.DataFrame({"F_stat":['{:.4e}'.format(float(F_stat))], "p_value":['{:.4e}'.format(float(p_val))]})
        self.table = table
        
        if prt:
            print("F-test statistic is", '{:.4e}'.format(float(F_stat)))
            print("F-test has df",q-p,",",n-q)
            print("p-value is", '{:.4e}'.format(float(p_val)))
            if p_val > sig_lev:
                print("under significance level",sig_lev,"cannot say submodel is not sufficient")
            else: #reject H0
                print("under significance level",sig_lev,"can say submodel is not sufficient")
            
            
    
    
    
    