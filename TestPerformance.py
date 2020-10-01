from sklearn import model_selection

class TestPerformance:
    
    def cross_validation(self, model, X, y, scoring, n=10, if_print=True):
        """
        model: sklearn model, e.g.from sklearn.linear_model import LogisticRegression, from sklearn.naive_bayes import GaussianNB, from sklearn.neighbors import KNeighborsClassifier
        scoring: measure method
            Classification: 'accuracy', 'average_precision', 'f1', 'f1_macro', 'f1_micro', 'f1_samples', 'f1_weighted', 'neg_log_loss',  'normalized_mutual_info_score', 'precision', 'precision_macro', 'precision_micro', 'precision_samples', 'precision_weighted', 'recall', 'recall_macro', 'recall_micro', 'recall_samples', 'recall_weighted', 'roc_auc'
            Clustering: 'adjusted_mutual_info_score', 'adjusted_rand_score', 'completeness_score','fowlkes_mallows_score', 'homogeneity_score', 'mutual_info_score', 'v_measure_score'
            Regression: 'explained_variance', 'neg_mean_absolute_error', 'neg_mean_squared_error', 'neg_mean_squared_log_error', 'neg_median_absolute_error', 'r2', 
        """
        name = type(model).__name__
        kfold = model_selection.KFold(n_splits=n)
        cv_results = model_selection.cross_val_score(model, X, y, cv=kfold, scoring=scoring)
        if if_print:
            print('Model: ' + name + ';', ' Measurement: ' + scoring + '; ', str(n) + '-fold')
            print('Mean = ' + str(cv_results.mean()) + '; ', 'Standard error = ' + str(cv_results.std()))
        return cv_results
    
    def compare_models_cross_validation(self, models, Xs, y, scoring, n=10):
        """
        model: list of sklearn models
        """
        results = []
        names = []
        for i, model in enumerate(models):
            name = type(model).__name__
            cv_results = self.cross_validation(model, Xs[i], y, scoring=scoring, if_print=False)
            results.append(cv_results)
            names.append(name)
            
        # boxplot algorithm comparison
        fig = plt.figure(figsize=(10,6))
        ax = fig.add_subplot(111)
        plt.boxplot(results)
        fig.suptitle('Algorithm Comparison by ' + scoring + ', ' + str(n) + '-fold')
        ax.set_xticklabels(names,rotation=25)
        plt.show()



