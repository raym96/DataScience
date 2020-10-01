import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.core.display import display
from scipy.stats import chi2_contingency
import glob
import os

import warnings
warnings.filterwarnings("ignore")

pd.options.display.float_format = '{:.4f}'.format

class DataExplorer():
    def load_data_from_folder(self, folderpath = "data/"):
        """
        Import all csv.file from the folder with given extension
        Returns the dictionary of pandas
        
        folderpath: example "data/"
        """
        my_dict = {}
        encodings = ['utf-8', 'cp1252','iso-8859-1','unicode_escape','ascii']
        for filename in glob.glob(os.path.join(folderpath, '*.csv')):
            filename = filename.replace("\\","/")
            loaded = False
            i = 0
            while loaded == False:
                if i == len(encodings):
                    print("[WARNING] The file named %s could not be loaded.\n" % filename)
                    break
                else:
                    try:
                        df = pd.read_csv(filename, encoding = encodings[i])
                        my_dict[filename.split(".")[0].split("/")[1]] = df
                        print("Filename:",filename.split(".")[0].split("/")[1],", loaded with encoding:",
                              encodings[i])
                        print("Shape:", df.shape)
                        display(df.head(5))
                        loaded = True
                        break
                    except:
                        i += 1
        return my_dict
    

    def describe(self, df, cols = None):
        """
        Data description (min, max, quantiles, missing data...)
        """
        if cols == None:
            cols = df.columns
        print(df.columns)
        print("\n Info ---------------------------")
        df[cols].info()
        # Description
        print("\n Description ---------------------------")
        description = df[cols].describe([0.1, 0.25, 0.75, 0.9], include = "all")
        description.loc['unique'] = df.nunique()
        display(description)
        # Unique values
        print("\n Unique values ---------------------------")
        display(df[cols].nunique())
        print("\n Missing ---------------------------")
        # Missing data
        total = df[cols].isnull().sum().sort_values(ascending=False)
        percent = (df[cols].isnull().sum()/df[cols].isnull().count()).sort_values(ascending=False)
        missing_data = pd.concat([total, percent], axis=1, keys=['N_MissingData', 'MissingDataPercent'])
        display(missing_data)
        
    def freq_analysis(self, df, cols = None, show_text = True):
        """
        Plot distribution for all columns
        show_text: If true, for categorical variables print the quantity 
        """
        if cols == None:
            cols = df.columns
        for col in cols:
            if np.issubdtype(df[col].dtype, np.number):
                sns.distplot(df[col].dropna(), kde = False)
                plt.title(col)
                plt.show()
            else:
                if len(df[col].unique()) <= 100: # Categorical varialbe with few categories
                    fig, ax = plt.subplots(figsize = (8,8))
                    count = df[col].value_counts(ascending = False)
                    count.iloc[::-1].plot(kind = 'barh')
                    total = count.sum()
                    cumsum = count.cumsum()
                    textstr = "N° of categories to represent 40%: {0:.0f} \n N° of categories to represent 60%: {1:.0f} \n N° categories to represent 80%: {2:.0f}".format( \
                        len(count.loc[:np.argmax(cumsum>total*0.4)]),len(count.loc[:np.argmax(cumsum>total*0.6)]),len(count.loc[:np.argmax(cumsum>total*0.8)]))
                    ax.text(0.3, 0.05, textstr, transform=ax.transAxes, fontsize=14,
                            verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='white', edgecolor = 'k',alpha=0.5))
                    if show_text:
                        for i, v in enumerate(count.iloc[::-1].values):
                            #ax.text(v+3, i-0.25, str(v),color='blue')
                            if count.nunique()>2:
                                ax.text(v+2, i-0.25, "{0:.1f}%".format(cumsum[-i-1]/total*100),color='blue')
                    plt.title(col)
                    plt.show()
    
    def correlation_analysis(self, df, cols = None, pairplot = False):
        """
        Visualize correlation between columns 
        Correlation pair plot between highest corr pairs
        """
        if cols == None:
            cols = df.columns
        corr = df[cols].corr()
        
        annot =  (len(corr) <= 15)
        
        # Only lower diag
        sns.set(style="white")
        mask = np.zeros_like(corr, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True

        f, ax = plt.subplots(figsize=(11, 9))
        cmap = sns.diverging_palette(220, 10, as_cmap=True)
        sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                    square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot = annot)
        ax.set_ylim(corr.shape[0]-1e-9, -1e-9)
        plt.show()
        
        # Another one
        f, ax = plt.subplots(figsize=(12, 9))
        sns.heatmap(corr, vmax=.8, square=True, annot = annot, linewidth = 0.2);
        ax.set_ylim(corr.shape[0]-1e-9, -1e-9)
        plt.show()
        
        # Cluster map
        clustergrid = sns.clustermap(corr, square = True, annot = annot, 
                                     linewidth = 0.2, figsize = (8,8), fmt = ".2f")
        clustergrid.ax_heatmap.set_ylim(corr.shape[0]-1e-9, -1e-9)
        plt.show()
        
        
        # Pair plot for highest correlation pairs
        if pairplot == True:
            corr_matrix = corr.abs()
            sol = (corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
                             .stack()
                             .sort_values(ascending=False))
            highest_corr_pair = sol.index.values[:3]
            highest_corr_cols = []
            for pair in highest_corr_pair:
                for x in pair:
                    if not x in highest_corr_cols:
                        highest_corr_cols.append(x)
            sns.pairplot(df[highest_corr_cols])
            plt.title("Highest correlation pairs")
            plt.show()
        
    def correlation_analysis_with_target(self, df, target, k = 10):
        """
        Correlation plot of columns that have the largest correlation with target 
        
        target: target column
        k: number of columns to consider, default 5
        """
        fig, ax = plt.subplots()
        corrmat = df.corr().abs()
        cols = corrmat.nlargest(k, target)[target].index
        cm = np.corrcoef(df[cols].values.T)
        sns.set(font_scale=1.25)
        hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
        ax.set_ylim(cm.shape[0]-1e-9, -1e-9)
        plt.show()
    
    def outlier_analysis(self, df, cols = None):
        """
        Box plot outliers
        """
        if cols == None:
            cols = df.columns
        for col in cols:
            if np.issubdtype(df[col].dtype, np.number) and not (df[col].isnull().values.any()):
                sns.boxplot(df[col])
                plt.title(col)
                plt.show()
        
    def boxplot(self, xcol, ycol, data):
        """
        xcol, ycol: string, column name
        Box plot
        """
        plt.subplots(figsize=(16, 8))
        sns.boxplot(x=xcol,y=ycol, data = data)
        plt.show()
        
    def pairplot(self, data, cols):
        """
        Correlation plot between all pairs of columns
        """
        sns.pairplot(data[cols])
    
    def catplot(self, x, hue, col, data):
        """
        x: value to count
        hue: color category
        col: type category
        
        Plot the count of categories
        """
        sns.catplot(x=x, hue = hue, col = col, data = data, kind = "count")
    
    def datetime_analysis(self, df, colname):
        """
        df: dataframe
        date_col is a STRING corresponding to a column name that represents a date
        
        Count plot all frequencies
        """
        date_df = df[[colname]].copy()
        date_df[colname] = pd.to_datetime(date_df[colname])
        date_df['YEAR'] = date_df[colname].dt.year
        date_df['MONTH'] = date_df[colname].dt.month
        date_df['DAY'] = date_df[colname].dt.day
        date_df['WEEKDAY'] = date_df[colname].dt.weekday
        date_df['HOUR'] = date_df[colname].dt.hour
        
        sns.countplot(x = "YEAR", data = date_df)
        plt.title("Year")
        plt.show()
        
        sns.countplot(x = "MONTH", data = date_df)
        plt.title("Month")
        plt.show()
        
        sns.countplot(x = "DAY", data = date_df)
        plt.title("Day")
        plt.show()
        
        sns.countplot(x = "WEEKDAY", data = date_df)
        plt.title("Weekday")
        plt.show()
        
        sns.countplot(x ="HOUR",  data = date_df)
        plt.title("Hour")
        plt.show()
        
        plt.subplots(figsize = (10,8))
        sns.countplot(x = "MONTH", hue = "YEAR", data = date_df)
        plt.title("Month - Yearly")
        plt.show()
        
        plt.subplots(figsize = (10,8))
        sns.countplot(x = "DAY", hue = "MONTH",  data = date_df)
        plt.title("Day - Monthly")
        plt.show()
        
        plt.subplots(figsize = (10,8))
        sns.countplot(x = "WEEKDAY", hue = "MONTH", data = date_df)
        plt.title("Weekday - Monthly")
        plt.show()
        
        sns.catplot(x = "WEEKDAY", hue = "MONTH", col = "YEAR", data = date_df, kind = "count")
        plt.savefig('week_month_year.png')
        
        sns.catplot(x = "HOUR", hue = "WEEKDAY", col = "MONTH", data = date_df, kind = "count")
        plt.savefig('hour_weekday_month.png')
        
        
    def map_categories(self, array, categories=None, ordered=False):
        """
        array: is 1-d data
        categories: list of unique catecories, will be encoded in this sequence
        """
        cat_type = pd.api.types.CategoricalDtype(categories=categories, ordered=ordered)
        return array.astype(cat_type).cat.codes
    
    def categorical_plot(self, data, category, func=None, value=None):
        """
        category: category column name (x axis)
        func: (Default None) count the number of different categories; (func: )max, mean, sum...
        value: when func is NOT None, should specify value (column name) to calculate (y axis).
        """
        if func is None:
            plt_data = data[category].value_counts()
            threshold = plt_data.mean() # mean of the counting data
            plt.figure(figsize=(10,6))
            plt.bar(list(plt_data.index), plt_data)          
            plt_data_over = plt_data
            plt_data_over[plt_data < threshold] = 0
            plt.bar(list(plt_data_over.index), plt_data_over , color='yellow')
            plt.plot(list(plt_data.index), [threshold] * len(plt_data), color='k', ls=':', lw=2)
            plt.text(plt_data.index.sort_values()[-2], threshold * 1.01,'Mean = '+ '{:.2f}'.format(threshold))
            plt.xticks(list(plt_data.index), rotation = 90)
            plt.title('Number of Observations in Category: ' + category)
            plt.show()
        else:
            if value is None:
                raise Exception("Should specify value column (continuous variable)")
            plt_data = data[[category, value]]
            plt_data = plt_data.groupby(category).apply(func)[value]
            threshold = plt_data.mean() # mean of the counting data
            plt.figure(figsize=(10,6))
            plt.bar(list(plt_data.index), plt_data)          
            plt_data_over = plt_data
            plt_data_over[plt_data < threshold] = 0
            plt.bar(list(plt_data_over.index), plt_data_over , color='yellow')
            plt.plot(list(plt_data.index), [threshold] * len(plt_data), color='k', ls=':', lw=2)
            plt.text(plt_data.index.sort_values()[-2], threshold * 1.01,'Mean = '+ '{:.2f}'.format(threshold))
            plt.xticks(list(plt_data.index), rotation = 90)
            plt.ylabel(value)
            plt.title(func.__name__.capitalize() + ' of ' + value + ' within Category: ' + category)
            plt.show()
    
    
    def distribution_plot(self, data, plot_variable, category_variable, hist = True, kde = False):
        """
        plot the distribution of 'plot_variable' for different categories of 'category_variable'
        
        plot_variable: continuous varialbe, xaxis
        category_variable: category variable, count on yaxis
        hist: True then plot histogram
        kde: true then plot density function
        """
        cats = data[category_variable].unique()
        plt.figure(figsize=(10,6))
        for cat in cats:
            subset = data[data[category_variable] == cat]
            sns.distplot(subset[plot_variable], hist = hist, kde = kde,
                         kde_kws = {'linewidth': 2, 'shade': False},label = cat)

        plt.grid(axis='x')
        plt.legend(loc='best')
        plt.xlabel(plot_variable)
        if hist and not kde:
            plt.ylabel('Histogram')
        if kde and not hist:
            plt.ylim((0,0.8))
            plt.ylabel('Density')
        if hist and kde:
            plt.ylabel('Histogram and Density')
        plt.show()
   
    def categorical_stack_bar(self, data, x_cat_name, y_cat_name, percentage = False):
        """
        x_cat_name: categorical column name that will be plot at x axis. Will be counted
        y_cat_name: categorical column name that will split the columns and color labelled
        """
        count_data = data[[x_cat_name, y_cat_name]]
        count_data['count'] = 0
        count_data = count_data.groupby([x_cat_name, y_cat_name]).count()
        count_data = count_data.pivot_table(index=x_cat_name, columns=y_cat_name, values='count')
        
        fig, ax = plt.subplots(figsize = (10,6))
        if percentage:
            count_data = count_data.divide(count_data.sum(axis=1), axis=0)
        count_data.plot.bar(stacked=True, ax = ax)
        vals = ax.get_yticks()
        ax.set_yticklabels(['{:,.0%}'.format(x) for x in vals])
        plt.show()

    
    # Basic stat tests
    def chi_2_test(self, df, column_A, column_B):
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
    
    
    