from sklearn.inspection import plot_partial_dependence
import matplotlib.pyplot as plt
import seaborn as sns


class Interpretation:

    def plot_pdp(self, learner, X, features_idx, feature_names=None):
        '''
        Plots the partial dependence plot for the given learner.
        Parameters:
        - learner: already trained learner to be analyzed
        - X: matrix of input data on which the learner has been trained
        - features_idx: features to be analyzed by pdp, should be column indexes in X
        - feature_names: features names of X
        '''
        fig = plt.figure(figsize=(20, 10))
        plot_partial_dependence(learner, X, features_idx, feature_names=feature_names, fig=fig)

    def plot_pdp_df(self, learner, df, str_y,  features_idx=None):
        '''
        Plots the partial dependence plot for the given learner and dataframe df.

        Parameters:
        - learner: already trained learner to be analyzed
        - df: dataframe considered
        - str_y: output variable
        - features_idx: features to be analyzed by pdp, should be a list of
                        column indexes in X
        '''
        if features_idx == None:
            features_idx = [i for i in range(len(df.columns)-1)]
        fig = plt.figure(figsize=(20, 10))
        X = df.loc[:, df.columns != str_y].values
        feature_names = df.columns[df.columns != str_y].to_list()

        df.columns[df.columns != str_y]
        plot_partial_dependence(learner, X, features_idx, feature_names=feature_names, fig=fig)
