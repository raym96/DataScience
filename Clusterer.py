import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D
import copy

from Utility import Utility
from FactorAnalyzer import FactorAnalyzer

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import fcluster

from sklearn import mixture

import itertools

def fancy_dendrogram(*args, **kwargs):
    max_d = kwargs.pop('max_d', None)
    if max_d and 'color_threshold' not in kwargs:
        kwargs['color_threshold'] = max_d
    annotate_above = kwargs.pop('annotate_above', 0)

    ddata = dendrogram(*args, **kwargs)

    if not kwargs.get('no_plot', False):
        plt.title('Hierarchical Clustering Dendrogram (truncated)')
        plt.xlabel('sample index or (cluster size)')
        plt.ylabel('distance')
        for i, d, c in zip(ddata['icoord'], ddata['dcoord'], ddata['color_list']):
            x = 0.5 * sum(i[1:3])
            y = d[1]
            if y > annotate_above:
                plt.plot(x, y, 'o', c=c)
                plt.annotate("%.3g" % y, (x, y), xytext=(0, -5),
                             textcoords='offset points',
                             va='top', ha='center')
        if max_d:
            plt.axhline(y=max_d, c='k')
    return ddata

class Clusterer:
    def silhouette_clusters_K_Means(self, X, range_n_clusters=[2,3,4,5,6]):
        '''Plots the silhouette coefficients values of the different clusters along with the average silhouette score
        Parameters:
        - X: data to be clustered of size (n_instances, n_features)
        - range_n_clusters: sizes of the clusters considered, for instance: range_n_clusters = [2, 3, 4, 5, 6]
        '''
        for n_clusters in range_n_clusters:
            # Create a subplot with 1 row and 2 columns
            fig, ax1 = plt.subplots(1, 1)
            #fig.set_size_inches(18, 7)

            # The 1st subplot is the silhouette plot
            # The silhouette coefficient can range from -1, 1 but in this example all
            # lie within [-0.1, 1]
            ax1.set_xlim([-1, 1])
            # The (n_clusters+1)*10 is for inserting blank space between silhouette
            # plots of individual clusters, to demarcate them clearly.
            ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

            # Initialize the clusterer with n_clusters value and a random generator
            # seed of 10 for reproducibility.
            clusterer = KMeans(n_clusters=n_clusters, random_state=10)
            cluster_labels = clusterer.fit_predict(X)

            # The silhouette_score gives the average value for all the samples.
            # This gives a perspective into the density and separation of the formed
            # clusters
            silhouette_avg = silhouette_score(X, cluster_labels)
            print("For n_clusters =", n_clusters,
                  "The average silhouette_score is :", silhouette_avg)

            # Compute the silhouette scores for each sample
            sample_silhouette_values = silhouette_samples(X, cluster_labels)

            y_lower = 10
            for i in range(n_clusters):
                # Aggregate the silhouette scores for samples belonging to
                # cluster i, and sort them
                ith_cluster_silhouette_values = \
                    sample_silhouette_values[cluster_labels == i]

                ith_cluster_silhouette_values.sort()

                size_cluster_i = ith_cluster_silhouette_values.shape[0]
                y_upper = y_lower + size_cluster_i

                color = cm.nipy_spectral(float(i) / n_clusters)
                ax1.fill_betweenx(np.arange(y_lower, y_upper),
                                  0, ith_cluster_silhouette_values,
                                  facecolor=color, edgecolor=color, alpha=0.7)

                # Label the silhouette plots with their cluster numbers at the middle
                ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

                # Compute the new y_lower for next plot
                y_lower = y_upper + 10  # 10 for the 0 samples

            ax1.set_title("The silhouette plot for " + str(n_clusters) + " clusters.")
            ax1.set_xlabel("The silhouette coefficient values")
            ax1.set_ylabel("Cluster label")

            # The vertical line for average silhouette score of all the values
            ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

            plt.show()

    def elbow_k_means(self, X, range_n_clusters = np.arange(2,11)):
        '''Plots the Elbow Method for K-Means
        Parameters:
        - X: data to be clustered of size (n_instances, n_features)
        - range_n_clusters: sizes of the clusters considered, for instance: range_n_clusters = [2, 3, 4, 5, 6]
        '''
        km = [KMeans(n_clusters=i, random_state=10) for i in range_n_clusters]
        inertias = [km[i].fit(X).inertia_ for i in range(len(km))]

        plt.plot(range_n_clusters, inertias, '.-', linewidth = 2)
        plt.xlabel("k: number of clusters")
        plt.ylabel("intertia")
        plt.title("Elbow method")
        return inertias
    
    def plot_cluster_centers(self, centers):
        """
        Heatmap visualization 
        """
        fig, ax = plt.subplots(figsize = (10,8))
        sns.heatmap(centers.T, annot = True, center = 0, fmt = '.2f', linewidth = 0.5, cmap = 'viridis')
        ax.set_title("Cluster centers", fontsize = 20)
        plt.xticks(rotation = 0)
        plt.yticks(rotation = 0)
        ax.set_ylim(centers.T.shape[0]-1e-9, -1e-9)
        ax.set_xlabel("Cluster")
        plt.show()
    
    
    def k_means(self, df, n_clusters):
        '''
        Returns the label encoded clusters for each example in the df.
        '''
        df = Utility().normalize(df)
        kmeans = KMeans(n_clusters = n_clusters)
        kmeans.fit(df)
        print("Reduced inertia:", kmeans.inertia_)
        print("Clusters centers:")
        display(pd.DataFrame(kmeans.cluster_centers_, columns = df.columns,
                             index = ["cluster %i" %i for i in np.arange(n_clusters)]))
        
        centers = pd.DataFrame(kmeans.cluster_centers_, columns = df.columns, index =["cluster %i" %i for i in np.arange(n_clusters)])
        self.plot_cluster_centers(centers)
        # return Utility().label_encode(kmeans.labels_)
        return kmeans.labels_

    def hierarchical_clusters(self, X, method='ward', p=30, k=4):
        '''Returns the clusters predicted for each example:
        - X: data to cluster
        - method: linkage method, by default 'ward'
        - p: last p splits considered in the dendogram
        - k: number of clusters used in the predicted clusters (does not influence the dendogram)
        '''

        Z = linkage(X, 'ward')

        plt.figure(figsize=(25, 10))

        _ = fancy_dendrogram(Z,
                             leaf_rotation=90.,  # rotates the x axis labels
                             leaf_font_size=8.,  # font size for the x axis labels
                             truncate_mode='lastp',
                             p=p
                             )
        
        clusters = np.array(fcluster(Z, k, criterion='maxclust')) -1
        X_copy = copy.deepcopy(X)
        X_copy['clusters'] = clusters
        centers = X_copy.groupby(['clusters']).mean()
        display(centers)
        self.plot_cluster_centers(centers)
        return clusters


    def best_GMM_clusters(self, X, criterion='bic'):
        '''
        Returns the best GMM clusters according to AIC or BIC and plots the criterion AIC and BIC
        found for each number of components used.
        Parameters:
        - X: data to be cluster of size (n_examples, n_features)
        - criterion: 'bic' or 'aic', criterion selected for the best gmm
        '''
        lowest_aic = np.infty
        lowest_bic = np.infty
        aic = []
        bic = []
        n_components_range = range(1, 7)
        cv_types = ['spherical', 'tied', 'diag', 'full']
        for cv_type in cv_types:
            for n_components in n_components_range:
                # Fit a Gaussian mixture with EM
                gmm = mixture.GaussianMixture(n_components=n_components,
                                              covariance_type=cv_type)
                gmm.fit(X)
                aic.append(gmm.aic(X))
                bic.append(gmm.bic(X))

                if aic[-1] < lowest_aic:
                    lowest_aic = aic[-1]
                    best_gmm_aic = gmm
                    best_type_aic = cv_type
                    best_component_aic = n_components
                if bic[-1] < lowest_bic:
                    lowest_bic = bic[-1]
                    best_gmm_bic = gmm
                    best_type_bic = cv_type
                    best_component_bic = n_components

        aic = np.array(aic)
        bic = np.array(bic)
        color_iter = itertools.cycle(['navy', 'turquoise', 'cornflowerblue',
                                      'darkorange'])
        bars = []
        plt.figure(figsize=(20, 8))
        # Plot the AIC scores
        spl = plt.subplot(1, 2, 1)
        for i, (cv_type, color) in enumerate(zip(cv_types, color_iter)):
            xpos = np.array(n_components_range) + .2 * (i - 2)
            bars.append(plt.bar(xpos, aic[i * len(n_components_range):
                                          (i + 1) * len(n_components_range)],
                                width=.2, color=color))
        plt.xticks(n_components_range)
        plt.ylim([aic.min() * 1.01 - .01 * aic.max(), aic.max()])
        plt.title('AIC score per model')
        xpos = np.mod(aic.argmin(), len(n_components_range)) + .65 +\
            .2 * np.floor(aic.argmin() / len(n_components_range))
        plt.text(xpos, aic.min() * 0.97 + .03 * aic.max(), '*', fontsize=14)
        spl.set_xlabel('Number of components')
        spl.legend([b[0] for b in bars], cv_types)
        # Plot the BIC scores
        spl = plt.subplot(1, 2, 2)
        for i, (cv_type, color) in enumerate(zip(cv_types, color_iter)):
            xpos = np.array(n_components_range) + .2 * (i - 2)
            bars.append(plt.bar(xpos, bic[i * len(n_components_range):
                                          (i + 1) * len(n_components_range)],
                                width=.2, color=color))
        plt.xticks(n_components_range)
        plt.ylim([bic.min() * 1.01 - .01 * bic.max(), bic.max()])
        plt.title('BIC score per model')
        xpos = np.mod(bic.argmin(), len(n_components_range)) + .65 +\
            .2 * np.floor(bic.argmin() / len(n_components_range))
        plt.text(xpos, bic.min() * 0.97 + .03 * bic.max(), '*', fontsize=14)
        spl.set_xlabel('Number of components')
        spl.legend([b[0] for b in bars], cv_types)

        if criterion == 'aic':
            # return Utility().label_encode(best_gmm_aic.predict(X))
            print('Criterion selected for clusters: AIC')
            print('Selected CV type:', best_type_aic)
            print('Selected number of components:', best_component_aic)
            return best_gmm_aic.predict(X)
        else:
            # return Utility().label_encode(best_gmm_bic.predict(X))
            print('Criterion selected for clusters: BIC')
            print('Selected CV type:', best_type_bic)
            print('Selected number of components:', best_component_bic)
            return best_gmm_bic.predict(X)

    def plot_cluster_2D(self, df, clusters, pca_plot = False):
        """
        Give a 2D view of clusters using 2-dimensional PCA

        df: dataframe
        clusters: list of labels
        """
        f = FactorAnalyzer()
        pca = f.pca(2, df, plot = pca_plot)
        palette = sns.color_palette('hls', len(np.unique(clusters))+1)

        plt.scatter(pca['results'][0],pca['results'][1], c = [palette[i] for i in clusters])

        handles = [mpatches.Patch(color=palette[i], label="cluster %i" % i) for i in np.unique(clusters)]
        plt.legend(handles = handles)
        plt.xlabel("PC1 ({:.2%} explained variance)".format(pca['explained_variance'][0]))    
        plt.ylabel("PC2 ({:.2%} explained variance)".format(pca['explained_variance'][1]))   
        plt.show()

    def plot_cluster_3D(self, df, clusters, pca_plot = False):
        """
        Give a 3D view of clusters using 3-dimensional PCA
        """
        f = FactorAnalyzer()
        pca = f.pca(3, df, plot = pca_plot)
        palette = sns.color_palette('hls', len(np.unique(clusters))+1)

        fig = plt.figure(figsize = (10,10))
        ax = fig.add_subplot(111, projection = '3d')

        ax.scatter(pca['results'][0], pca['results'][1], pca['results'][2],
                   c = [palette[i] for i in clusters], marker = 'o', s = 50)
        ax.set_xlabel("PC1 ({:.2%} explained variance)".format(pca['explained_variance'][0]))    
        ax.set_ylabel("PC2 ({:.2%} explained variance)".format(pca['explained_variance'][1]))   
        ax.set_zlabel("PC3 ({:.2%} explained variance)".format(pca['explained_variance'][2]))   

        scatter_proxy = [Line2D([0],[0], linestyle = 'none', c = palette[i], marker = 'o') for i in np.unique(clusters)]
        label = ["cluster %i" % i for i in np.unique(clusters)]
        ax.legend(scatter_proxy, label, loc = (0.7,0.6))
        fig.show()
