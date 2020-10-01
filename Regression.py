import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
pd.set_option('display.width', 100)
pd.set_option('precision', 3)

import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from sklearn.metrics import mean_squared_error, r2_score
from linearmodels import PanelOLS

class linear_regression:
    def __init__(self):
        self.reg = None
    
    def ols_fit(self, X, y, add_const=True, plot=None, assumption_test=False, if_print=True):
        """
        plot: column name of X that want to plot
        """
        if add_const:
            X_ = sm.add_constant(X)
        mod = sm.OLS(y.values, np.asarray(X_))
        res = mod.fit()        
        #print(res.summary())
        res = res.get_robustcov_results()
        self.reg = res
        if if_print:
            print(res.summary()) # Robusted Results
        
        if plot is not None:
            prstd, iv_l, iv_u = wls_prediction_std(self.reg)
            plt.figure(figsize=(10,6))
            plt.plot(X[plot], y, 'o', label='Sample Data')
            fit_data = pd.DataFrame({'X': X[plot],
                                     'y': res.fittedvalues,
                                    'upper': iv_u,
                                    'lower': iv_l})
            fit_data.sort_values(by='X', inplace=True)
            plt.plot(fit_data.X, fit_data.y, 'r', label='OLS model')
            plt.plot(fit_data.X, fit_data.upper, color='darksalmon', ls='--')
            plt.plot(fit_data.X, fit_data.lower, color='darksalmon', ls='--')
            plt.legend(loc='best')
            plt.show()
            
        if assumption_test:
            error = res.fittedvalues - y
            print('Residual Tests:')
            # Durbin-Waston test: 
            dw = durbin_watson(error)
            print('DW test statistic: ', dw)
            print('(2: no serial correlation; 0: positive serial correlation; 4: negative serial correlation)')
            
            sm.graphics.tsa.plot_acf(error, lags=20) # residual auto-correlation
            
            sm.qqplot(error, loc = error.mean(), scale = error.std(), line='s') # residual normality
            
    def gls_fit(self, X, y, add_const=True):
        if add_const:
            X_ = sm.add_constant(X)
        mod = sm.GLS(y.values, np.asarray(X_))
        res = mod.fit()        
        #print(res.summary())
        res = res.get_robustcov_results()
        self.reg = res
        print(res.summary()) # Robusted Results
    
    def get_predict(self, X_test, add_const=True):
        if add_const:
            X_test = sm.add_constant(X_test)
        return self.reg.predict(X_test)
    
    def test_performance(self, X_test, y_test, add_const=True, plot=None):
        y_pred = self.get_predict(X_test, add_const=add_const)
        print("Mean squared error: %.2f"% mean_squared_error(y_test, y_pred))
        # Explained variance score: 1 is perfect prediction
        print('R-2 score: %.2f' % r2_score(y_test, y_pred))
        
        if plot is not None:
            plt.figure(figsize=(10,6))
            plt.scatter(X_test[plot], y_test, color='black', label='Test Sample Data')
            fit_data = pd.DataFrame({'X': X_test[plot],
                                     'y': y_pred})
            fit_data.sort_values(by='X', inplace=True)
            plt.plot(fit_data.X, fit_data.y, color='blue',label='Model Value', linewidth=3)
            plt.legend(loc='best')
            plt.show()
        
    def get_parameters(self, pr=True):
        """
        get parameters, standard errors, t-values, p-values
        """
        if pr:
            print('Coeficients: ', self.reg.params)
            print('Standard error: ', self.reg.bse)
            print('T-values: ', self.reg.tvalues) 
            print('p-values: ', self.reg.pvalues) 
        return self.reg.params, self.reg.bse, self.reg.tvalues, self.reg.pvalues
    
    def get_confidence_intervals(self, alpha=0.05):
        return self.reg.conf_int(alpha=alpha)
    
    def print_latex(self):
        return self.reg.summary().as_latex()
    
    def get_residuals(self, normalize=False):
        """
        normalize: Default False, return normalized residuals to have unit variance if True
        """
        if normalize:
            return self.reg.resid_pearson
        else:
            return self.reg.resid
    
    def get_r_square(self):
        """
        return R2 and Adj-R2
        """
        return self.reg.rsquared, self.reg.rsquared_adj
        
    def panel_regression(self, X, y, entity_col, time_col, entity_effects=False, time_effects=False, other_effects=None, add_const=True, drop_absorbed=True):
        """
        other_effects (array-like) – Category codes to use for any effects that are not entity or time effects. Each variable is treated as an effect
        return fitted res
        """

        X = X.set_index([entity_col, time_col])
        y.index = X.index
        if add_const:
            X = sm.add_constant(X)
        if other_effects is None:
            mod = PanelOLS(y, X, entity_effects=entity_effects, time_effects=time_effects)#, endog_names=['intercept'] + X.columns)
        else:
            mod = PanelOLS(y, X, entity_effects=entity_effects, time_effects=time_effects, other_effects=X[other_effects])
        res = mod.fit()
        print(res.summary)
        return res
       
    def glm_regression(self, X, y, mod_family, add_const=True):
        """
        mod_family: sm.families.Binomial([link]), Gamma(), Gaussian(),InverseGaussian(), NegativeBinomial(), Poisson(), Tweedie()
        link: CDFlink, CLogLog, Log, Logit, NegativeBinomial([alpha]), Power([power]), cauthy(), cloglog, identity(), inverse_power(), inverse_sqared(), log, logit, nbinom([alphal]), probit([dbn])
        return fitted res
        """
        if add_const:
            X = sm.add_constant(X)
        mod = sm.GLM(y, X, family=mod_family)
        res = mod.fit()
        print(res.summary())
        return res

import statsmodels.api as sm
from scipy.special import expit

class logistic_regression:
    def __init__(self):
        self.reg = None
        
    def fit(self, X, y, add_const=True, plot=None, method='bfgs', if_print=True):
        """
        print summary table, prediction table (pred_table[i,j] refers to the number of times “i” was observed and the model predicted “j”. Correct predictions are along the diagonal.)
        """
        if add_const:
            X_ = sm.add_constant(X)
        mod = sm.Logit(y, X_)
        res = mod.fit(method=method)
        self.reg = res
        if if_print:
            print(res.summary())
            print('Prediction Table:')
            print(pd.DataFrame(res.pred_table()))
        
        if plot is not None:
            plt.clf()
            plt.figure(figsize=(10,6))
            plt.scatter(X[y == 1][plot], y[y == 1], color='r',label='Positive Sample', zorder=1)
            plt.scatter(X[y == 0][plot], y[y == 0], color='b',label='Negative Sample', zorder=1)
            X_ = X_.sort_values(by=plot)
            logit = expit(np.dot(X_, res.params))
            plt.plot(X_[plot], logit, 'k--',label='Fitted Model', linewidth=1)
            plt.legend(loc='best')
            plt.show()            
            
    def get_predict(self, X_test, add_const=True):
        if add_const:
            X_test_ = sm.add_constant(X_test)       
        return self.reg.predict(X_test_) 
        
    def test_performance(self, X_test, y_test, plot=None, add_const=True):
        """
        print out precision, accuracy, recall, f1
        """
        y_pred = self.get_predict(X_test, add_const=add_const)
        tp = np.count_nonzero(np.all([y_pred == 1, y_test == 1], axis=0))
        fp = np.count_nonzero(np.all([y_pred == 1, y_test == 0], axis=0))
        tn = np.count_nonzero(np.all([y_pred == 0, y_test == 0], axis=0))
        fn = np.count_nonzero(np.all([y_pred == 0, y_test == 1], axis=0))
        if tp+fp > 0:
            precision = tp / (tp+fp)
            print('Precision: ', precision)
        if tp+fn > 0:
            recall = tp / (tp + fn)
            print('Recall: ', recall)
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        print('Accuracy: ', accuracy)
        if tp+fp > 0 and tp + fn >0:
            print('F-measure: ', 2 * precision * recall / (precision + recall))
        
        if plot is not None:
            plt.clf()
            plt.figure(figsize=(10,6))
            plt.scatter(X_test[y_test == 1][plot], y_test[y_test == 1], color='r',label='Positive Sample', zorder=1)
            plt.scatter(X_test[y_test == 0][plot], y_test[y_test == 0], color='b',label='Negative Sample', zorder=1)
            if add_const:
                X_test_ = sm.add_constant(X_test)
            X_test_ = X_test_.sort_values(by=plot)
            logit = expit(np.dot(X_test_, self.reg.params))
            plt.plot(X_test_[plot], logit, 'k--',label='Fitted Model', linewidth=1)
            plt.legend(loc='best')
            plt.show()
            
    def get_coefficients(self):
        return self.reg.params
    
    def get_confidence_intervals(self, alpha=0.05):
        return self.reg.conf_int(alpha=alpha)
    
    def print_latex(self):
        return self.reg.summary().as_latex()
    
    def get_r_square(self):
        """
        McFadden’s pseudo-R-squared
        """
        return self.reg.prsquared