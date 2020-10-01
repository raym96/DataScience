import numpy as np
import pandas as pd

from scipy.stats import chi2_contingency


class Tests:

    def chi_2_test(column_A, column_B, df):
        '''
        Chi-square test of independence of categorical column A and column B of the dataframe df.
        '''
        obs = pd.crosstab(df[column_A], df[column_B])
        chi2, p, dof, ex = chi2_contingency(obs, correction=False)
        print("Chi2 test:")
        print("Chi2:", chi2, "p value:", p, "dof:", dof)
        # Significance test at level 0.05:
        if p < 0.05:
            print("p<0.05: Null hypothesis of independence is rejected")
        else:
            print("p>=0.05: Null hypothesis of independence is not rejected")