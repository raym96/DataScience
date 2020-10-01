import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
from itertools import cycle

class Lasso:
    def __init__(self):
        self.res = None
        self.type = None
        self.alpha = None
    
    # lasso with parameter selected using cross validation
    # tail_num is the first and last number of coefficients to plot
    def Lasso_cv(self, X, y, n_alps=100, alps=None, plot=True, tail_num = 5):
        X['intercept'] = 1
        lasso = linear_model.LassoCV(n_alphas = n_alps, alphas = alps)
        self.type = 'LassoCV'
        res = lasso.fit(X,y)
        self.res = res
        coefs = pd.Series(res.coef_, index = X.columns)
        print("Lasso picked " + str(sum(coefs != 0)) + " variables and eliminated the other " +  str(sum(coefs == 0)) + " variables")
        # get residuals 
        preds = pd.DataFrame({"preds":self.res.predict(X), "true":y})
        preds["residuals"] = preds["true"] - preds["preds"]
        # get best alpha
        self.alpha = res.alpha_
        print("Lasso picked the best penalty parameter",self.alpha)
        # plot results if plot==True
        if plot:
            imp_coef = pd.concat([coefs.sort_values().head(tail_num),
                         coefs.sort_values().tail(tail_num)])
            plt.figure(figsize=(8, 6), dpi=100)
            imp_coef.plot(kind = "barh")
            plt.title("Coefficients in the Lasso Model")
            plt.show()
            # plot residuals
            plt.figure(figsize=(8, 6), dpi=100)
            preds.plot(x = "true", y = "residuals",kind = "scatter")
            plt.title("True values versus residuals")
            plt.show()
    
    # compute lasso path and plot coefficients along the path
    def Lasso_path(self, X, y, alphs=None):
        X['intercept'] = 1
        alphs, path, _ = linear_model.lasso_path(X, y, alphas=alphs)
        self.type = 'Lasso_path'
        self.res = path
        coefs = path
        plt.figure(figsize=(8, 6), dpi=100)
        colors = cycle(['b', 'r', 'g', 'c', 'k'])
        for coef_l, c in zip(coefs, colors):
            plt.plot(-np.log(alphs), coef_l, c=c)
        plt.xlabel('-Log(alpha)')
        plt.ylabel('coefficients')
        plt.title('Coefficients along the path of Lasso')
        plt.axis('tight')
        plt.show()
    
    # plain lasso
    def plain_Lasso(self, X, y, alp, plot=True, tail_num=5):
        X['intercept'] = 1
        lasso = linear_model.Lasso(alpha=alp)
        self.res = lasso.fit(X, y)
        # get coefficients
        coefs = pd.Series(self.res.coef_, index = X.columns)
        print("Lasso picked " + str(sum(coefs != 0)) + " variables and eliminated the other " +  str(sum(coefs == 0)) + " variables")
        # get residuals 
        preds = pd.DataFrame({"preds":self.res.predict(X), "true":y})
        preds["residuals"] = preds["true"] - preds["preds"]
        # plot results if plot==True
        if plot:
            imp_coef = pd.concat([coefs.sort_values().head(tail_num),
                         coefs.sort_values().tail(tail_num)])
            plt.figure(figsize=(8, 6), dpi=100)
            imp_coef.plot(kind = "barh")
            plt.title("Coefficients in the Lasso Model")
            plt.show()
            # plot residuals
            plt.figure(figsize=(8, 6), dpi=100)
            preds.plot(x = "true", y = "residuals",kind = "scatter")
            plt.title("True values versus residuals")
            plt.show()