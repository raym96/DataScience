
import numpy as np
import pandas as pd
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
from math import ceil

class ResidualAnalysis:
    def __init__(self, data):
        self.data = data
        self.data.dropna(inplace=True)
        
    def fit_first_step_regression(self, y_col, x_cols, x_cat_cols=[], as_latex=True, cmap='Reds'):
        """
        Fit the first stage regression; print the regression summary; add residuals, reg, predict as a class property。
        y_col: (string) column name for y
        x_cols: (list of strings) column names for numerical type x
        x_cat_cols: (list of strings) column names for categorical type x
        """
        if len(x_cat_cols) == 0:
            cat_formula = ''
        else:
            cat_formula = ' + C(' + ') + C('.join(x_cat_cols) + ')'
        
        colors = dict(zip(df_map.index, cmap(labelcol_normalized)))
        colormap = []
        formula = y_col + ' ~ ' + ' + '.join(x_cols) +  cat_formula
        ols_first = ols(data = self.data, formula =formula).fit()
        print(ols_first.summary())
        if as_latex:
            print(ols_first.summary().as_latex())
        self.residuals = ols_first.resid
        self.reg = ols_first
        self.predict = ols_first.predict()
        
    def residual_plots_y(self):
        plt.figure(figsize=(10,6))
        plt.scatter(self.predict, self.residuals, c=self.residuals**2)
        plt.axhline(y=0, color='r', ls='--')
        plt.xlabel('Predicted Value')
        plt.ylabel('Residual')
        plt.show()
    
    def residual_plots_x(self, plot_x_cols):
        n = len(plot_x_cols)
        plt.figure(figsize=(15, 6*ceil(n/2)))
        for i in range(n):
            plt.subplot(ceil(n/2), 2, i+1)
            plt.scatter(self.data[plot_x_cols[i]], self.residuals, c=self.residuals**2)
            plt.axhline(y=0, color='r', ls='--')
            plt.xlabel(plot_x_cols[i])
            plt.ylabel('Residual')
        plt.show()


