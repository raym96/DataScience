import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
from Regression import linear_regression
help(linear_regression)
lr = linear_regression()

class Robustness:
        
    def stars(self, p):
        if p <= 0.001:
            return '***'
        elif p <= 0.05:
            return '**'
        elif p <= 0.1:
            return '*'
        else:
            return ''
    
    def double_sort(self, X, y, group_names, ngroup=5, take_in_reg = False):
        """
        X: contains cat_names
        take_in_reg: whether take the group_names into regression or not, Default False -> treate like traditional Fama Model alpha 
        group_names: list of two strings, first name will be show on the index, second name will be show on the column
        sort the regression residuals by two cat_names, compare n (biggest) vs 1 (smallest) group by t-test
        """
        X_cols = list(X.columns)
        if not take_in_reg:
            for group in group_names:
                X_cols.remove(group)
        lr.ols_fit(X[X_cols], y, if_print=False)
        resid = lr.get_residuals()
        XX = pd.concat([X[group_names], pd.Series(resid, name='residual', index=X.index)], axis=1)
        for group in group_names:
            XX[group + '_group'] = pd.qcut(XX[group].rank(method='first'), ngroup, labels=False) + 1 # 1-smallest, 5-biggist
        ds_df = pd.pivot_table(XX, values='residual', columns=group_names[1] + '_group', index=group_names[0] + '_group',aggfunc='mean')
        
        test_0 = ds_df.loc[5,:] - ds_df.loc[1,:] # test for first group_name, will add as the last raw
        test_1 = ds_df.loc[:,5] - ds_df.loc[:, 1] # test for first group_name, will add as the last column       
        
        XX_group = XX.groupby([group+'_group' for group in group_names])
        test_0_stars = ["{:.4f}".format(test_0[i]) + self.stars(ttest_ind(XX_group.get_group((1, i))['residual'], XX_group.get_group((5, i))['residual'])[1]) for i in range(1,6)] 
        test_1_stars = ["{:.4f}".format(test_1[i]) + self.stars(ttest_ind(XX_group.get_group((i, 1))['residual'], XX_group.get_group((i, 5))['residual'])[1]) for i in range(1,6)]
        
        ds_df = pd.concat([ds_df, pd.DataFrame({group_names[0] + ' (5-1)':test_0_stars}, index=ds_df.columns).T], axis=0)
        ds_df = pd.concat([ds_df, pd.DataFrame({group_names[1] + ' (5-1)':test_1_stars}, index=ds_df.columns)], axis=1)
        ds_df = ds_df.rename(index={1: '1 (smallest)', 5: '5 (biggest)'}, columns={1: '1 (smallest)', 5: '5 (biggest)'})
        return ds_df        

    
    def regression_result(self, X, y):          
        lr.ols_fit(X, y, if_print=False)
        pa, s, t, pv = lr.get_parameters(pr=False)
        res = pd.DataFrame({'paras': pa, 'pvalues': pv}, index=['intercept'] + list(X.columns))
        res['paras'] = res.apply(lambda x: "{:.4f}".format(x['paras']) + self.stars(x['pvalues']), axis=1)
        r2, ar2 = lr.get_r_square()
        res_r2 = pd.Series([r2, ar2], index=['R2', 'Adj-R2'], name='paras')
        res = pd.concat([res['paras'], res_r2], sort=False, axis=0)
        return res
        
    def cross_effects(self, X, y, keyvar_name, dummy_names):
        """
        X: contains all the dummys
        for the key variate, test if there exists cross effect for different dummies
        """
        X_cols_orig = list(X.columns)
        X_cols = [keyvar_name]
        X_cols_others = X_cols_orig.copy()
        X_cols_others.remove(keyvar_name)
        for dummy in dummy_names:
            X_cols_others.remove(dummy)
        res_all = pd.DataFrame([])
        
        res = self.regression_result(X[X_cols_others + [keyvar_name]], y)
        res = res.rename('Base')
        res_all = pd.concat([res_all, res], sort=False, axis=1)
        
        for dummy in dummy_names:
            cross_name = keyvar_name + ' x ' + dummy
            X[cross_name] = X[keyvar_name] * X[dummy]
            X_cols = X_cols + [dummy, cross_name]
            res = self.regression_result(X[X_cols_others + [keyvar_name, dummy, cross_name]], y)
            res = res.rename('ADD: ' + dummy)
            res_all = pd.concat([res_all, res], sort=False, axis=1)
        return res_all.loc[X_cols + X_cols_others + ['intercept', 'R2', 'Adj-R2']]       
    
    
    def stepwise_regression(self, X, y, X_cols):
        """
        conduct stepwise tests for X in sequence X_cols
        """    
        print('Stepwise test for ' + y.name + ': ' + '-> '.join(X_cols))
        for i in range(len(X_cols)):
            reg_X_cols = X_cols[:i+1]
            res = self.regression_result(X[reg_X_cols], y)
            res = res.rename(i+1)
            res_all = pd.concat([res_all, res], sort=False, axis=1)
        return res_all.loc[reg_X_cols + ['intercept', 'R2', 'Adj-R2']]


