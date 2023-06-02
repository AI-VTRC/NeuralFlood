from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import numpy as np


class OrderedClustering:
    """
    A class for performing ordered clustering analysis on a given data column.

    Args:
        col (pandas.Series): The data column to perform clustering on.
        max_iter (int): The maximum number of clusters to consider. Default is 10.
        n_init (int): The number of times the k-means algorithm will be run with different centroid seeds.
                      The final results will be the best output of n_init consecutive runs in terms of inertia.
                      Default is 100.

    Methods:
        clustering(): Perform the clustering analysis on the data column.
        elbow_diagram(inertia_results): Plot the elbow diagram to visualize the inertia values.
        plot_clusters(clusters): Plot the 1D clustering visualization of the clusters.
        relabeling(cluster): Relabel the data column based on the cluster results.

    Usage:
        ordered_clustering = OrderedClustering(col)
        cluster_results, inertia_results = ordered_clustering.clustering()
        ordered_clustering.elbow_diagram(inertia_results)
        ordered_clustering.plot_clusters(cluster_results)
        relabeled_col = ordered_clustering.relabeling(cluster_results)

    """

    def __init__(self, col, max_iter=10, n_init=100):
        """
        Initialize the OrderedClustering object.

        Args:
            col (pandas.Series): The data column to perform clustering on.
            max_iter (int): The maximum number of clusters to consider. Default is 10.
            n_init (int): The number of times the k-means algorithm will be run with different centroid seeds.
                          The final results will be the best output of n_init consecutive runs in terms of inertia.
                          Default is 100.
        """
        self.max_iter = max_iter
        self.n_init = n_init
        self.col = col

    def clustering(self):
        """
        Perform clustering analysis on the data column.

        Returns:
            tuple: A tuple containing the cluster results and the inertia values.
                   - cluster_results (list): List of clusters, where each cluster contains its label and corresponding data points.
                   - inertia_results (list): List of inertia values for different numbers of clusters.
        """
        # Extract the column as a NumPy array
        data = self.col.values.reshape(-1, 1)

        # Initialize a range of values for the number of clusters
        clusters_range = range(1, self.max_iter + 1)

        # Create an empty list to store the inertia values
        inertia_results = []

        # List to store cluster results
        cluster_results = []

        print("----------------------------------------")

        # Iterate over each value of the number of clusters
        for n_clusters_val in clusters_range:
            # Perform k-means clustering with the specified number of clusters and minimize within-cluster variance
            kmeans = KMeans(n_clusters=n_clusters_val, n_init=self.n_init).fit(data)

            # Calculate the inertia value (within-cluster sum of squares)
            inertia = kmeans.inertia_

            # Get the cluster centers and sort them based on ascending means
            cluster_centers = kmeans.cluster_centers_
            sorted_indices = np.argsort(cluster_centers.flatten())
            sorted_centers = cluster_centers[sorted_indices]

            # Get the cluster sizes
            labels = kmeans.labels_
            cluster_sizes = np.bincount(labels)

            # Create a dictionary to store the data points for each cluster
            clusters = {i: [] for i in range(n_clusters_val)}

            # Assign data points to the corresponding cluster
            for i, label in enumerate(labels):
                clusters[label].append(data[i])

            # Sort the clusters based on the sorted cluster centers
            sorted_clusters = [(i + 1, np.unique(clusters[sorted_indices[i]])) for i in range(n_clusters_val)]

            # Append the inertia value to the list
            inertia_results.append(inertia)

            # Print the information for the current number of clusters
            print(f"Number of Clusters: {n_clusters_val}")
            print(f"\nInertia: {inertia}")

            # Calculate the silhouette score if there are at least two unique labels
            if len(np.unique(labels)) > 1:
                silhouette = silhouette_score(data, labels)
                print(f"\nSilhouette Score: {silhouette}")
            else:
                print("\nSilhouette Score: 0 (Only one unique label)")

            # Print the sorted cluster centers and relabeled clusters
            print(f"\nCluster Centers (sorted by ascending means):")
            for i, center in enumerate(sorted_centers):
                print(f"Cluster {i + 1}: {center[0]}")

            # Print the sorted and relabeled clusters
            print(f"\nSorted and Relabeled Clusters (unique points):")
            for label, cluster in sorted_clusters:
                print(f"Cluster {label}: {list(cluster)}")

            # Print the cluster sizes
            print(f"\nNumber of Points in Clusters:")
            for i, (label, cluster) in enumerate(sorted_clusters):
                print(f"Cluster {label}: {len(clusters[sorted_indices[i]])}")

            print("----------------------------------------")

            cluster_results.append(sorted_clusters)

        return cluster_results, inertia_results

    def elbow_diagram(self, inertia_results):
        """
        Plot the elbow diagram to visualize the inertia values.

        Args:
            inertia_results (list): List of inertia values for different numbers of clusters.
        """
        plt.plot(range(1, self.max_iter + 1), inertia_results, marker='o')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Inertia')
        plt.title('Elbow Diagram')
        plt.show()

    def plot_clusters(self, clusters):
        """
        Plot the 1D clustering visualization of the clusters.

        Args:
            clusters (list): List of clusters, where each cluster contains its label and corresponding data points.
        """
        plt.figure(figsize=(8, 6))
        for i, (_, cluster) in enumerate(clusters):
            plt.scatter(cluster, np.zeros_like(cluster), label=f"Cluster {i + 1}")
        plt.xlabel('Value')
        plt.title('1D Clustering Visualization')
        plt.legend()
        plt.gca().axes.get_yaxis().set_visible(False)  # Remove y-axis scale
        plt.show()

    def relabel(self, cluster):
        """
        Relabel the data column based on the cluster results.

        Args:
            cluster (list): List of clusters, where each cluster contains its label and corresponding data points.

        Returns:
            list: The relabeled data column.
        """
        relabeled_col = []

        for cell in self.col:
            for cluster_label, cluster_values in cluster:
                if cell in cluster_values:
                    relabeled_col.append(cluster_label)
                    break

        return relabeled_col
