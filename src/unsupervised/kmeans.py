# -*- coding: utf-8 -*-

import heapq
import numpy as np
from sklearn.cluster import KMeans


class CustomKMeans(KMeans):
    y_train = None

    def __init__(self, n_clusters=8, init='k-means++', n_init=10, max_iter=300,
                 tol=1e-4, precompute_distances='auto',
                 verbose=0, random_state=None, copy_x=True, n_jobs=1, n_suggested_tags=2):
        super(CustomKMeans, self).__init__(n_clusters=n_clusters, init=init, n_init=n_init, max_iter=max_iter,
                 tol=tol, precompute_distances=precompute_distances,
                 verbose=verbose, random_state=random_state, copy_x=copy_x, n_jobs=n_jobs)
        self.n_suggested_tags = n_suggested_tags


    def predict(self, X):
        y_predicted_clusters = super(CustomKMeans, self).predict(X)
        y_predicted_classes = []

        for y_predicted_cluster in y_predicted_clusters:
            y = [0]
            for idx, label in enumerate(self.labels_):
                if label == y_predicted_cluster:
                    y += self.y_train[idx]
            largest_values = heapq.nlargest(self.n_suggested_tags, y)
            n_tag_assignments = 0

            for idx, tag_assignments in enumerate(y):
                if tag_assignments > 1 and tag_assignments in largest_values \
                and n_tag_assignments < self.n_suggested_tags:
                    y[idx] = 1
                    n_tag_assignments += 1
                else:
                    y[idx] = 0

            #assert sum(y) == n_tags
            y_predicted_classes.append(y.tolist())
        return np.array(y_predicted_classes)


    def fit(self, X, y=None):
        self.y_train = y
        return super(CustomKMeans, self).fit(X, y)
