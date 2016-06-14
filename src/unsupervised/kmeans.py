# -*- coding: utf-8 -*-

import heapq
import numpy as np
from scipy.sparse import vstack
from sklearn.cluster import KMeans


class CustomKMeans(KMeans):
    x_train = None
    y_train = None

    def __init__(self, n_suggested_tags, n_clusters=8, init='k-means++', n_init=10, max_iter=300,
                 tol=1e-4, precompute_distances='auto',
                 verbose=0, random_state=None, copy_x=True, n_jobs=1):
        super(CustomKMeans, self).__init__(n_clusters=n_clusters, init=init, n_init=n_init, max_iter=max_iter,
                 tol=tol, precompute_distances=precompute_distances,
                 verbose=verbose, random_state=random_state, copy_x=copy_x, n_jobs=n_jobs)

        self.n_suggested_tags = n_suggested_tags


    def predict(self, X=None):
        X_complete = vstack([self.x_train, X])
        super(CustomKMeans, self).fit(X_complete.toarray())

        assert len(self.labels_) == X_complete.shape[0]

        y_predicted_clusters = self.labels_[self.x_train.shape[0]:]
        y_predicted_classes = []

        assert len(y_predicted_clusters) == X.shape[0]

        for y_predicted_cluster in y_predicted_clusters:
            y = np.array([0] * self.y_train.shape[1])
            for idx, y_train_data in enumerate(self.y_train):
                if self.labels_[idx] == y_predicted_cluster:
                    y += y_train_data
            largest_values = heapq.nlargest(self.n_suggested_tags, y)
            n_tag_assignments = 0

            for idx, tag_data in enumerate(y):
                if tag_data in largest_values and n_tag_assignments < self.n_suggested_tags:
                    y[idx] = 1
                    n_tag_assignments += 1
                else:
                    y[idx] = 0

            assert sum(y) == self.n_suggested_tags
            y_predicted_classes.append(y)
        return np.array(y_predicted_classes)

    def fit(self, X, y=None):
        self.x_train = X
        self.y_train = y
        return self
