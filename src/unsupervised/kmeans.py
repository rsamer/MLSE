# -*- coding: utf-8 -*-

import heapq
import numpy as np
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
        assert len(self.labels_) == self.x_train.shape[0]
        y_predicted_classes = []
        y_predicted_clusters = super(CustomKMeans, self).predict(X.toarray())
        assert len(y_predicted_clusters) == X.shape[0]

        # predict top X most frequent intra-cluster tags for each test post
        for y_predicted_cluster in y_predicted_clusters:
            y = np.array([0] * self.y_train.shape[1])  # store inter-cluster occurrence for each tag

            for post_idx, y_train_data in enumerate(self.y_train):
                if self.labels_[post_idx] == y_predicted_cluster:  # train and test data are in same cluster
                    y += y_train_data

            # predict most frequent tags (= tags with highest tag occurrences)
            largest_values = heapq.nlargest(self.n_suggested_tags, y)
            n_tag_assignments = 0

            for tag_idx, tag_data in enumerate(y):
                if tag_data in largest_values and n_tag_assignments < self.n_suggested_tags:
                    y[tag_idx] = 1  # this post belongs to this tag
                    n_tag_assignments += 1
                else:
                    y[tag_idx] = 0  # this post does not belong to this tag

            assert sum(y) == self.n_suggested_tags
            y_predicted_classes.append(y)

        return np.array(y_predicted_classes)

    def fit(self, X, y=None):
        self.x_train = X
        self.y_train = y
        return super(CustomKMeans, self).fit(X, y)
