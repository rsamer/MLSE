# -*- coding: utf-8 -*-

import heapq
import numpy as np
from scipy.sparse import vstack
from sklearn.externals.joblib import Memory
from sklearn.cluster import AgglomerativeClustering


class CustomHAC(AgglomerativeClustering):
    x_train = None
    y_train = None

    def __init__(self, n_suggested_tags, n_clusters=2, affinity="euclidean",
                 memory=Memory(cachedir=None, verbose=0),
                 connectivity=None, n_components=None,
                 compute_full_tree='auto', linkage='ward',
                 pooling_func=np.mean, ):
        super(CustomHAC, self).__init__(n_clusters=n_clusters, affinity=affinity, memory=memory,
                                        connectivity=connectivity, n_components=n_components,
                                        compute_full_tree=compute_full_tree, linkage=linkage,
                                        pooling_func=pooling_func)

        self.n_suggested_tags = n_suggested_tags

    def predict(self, X=None):
        X_complete = vstack([self.x_train, X])
        super(CustomHAC, self).fit(X_complete.toarray())
        assert len(self.labels_) == X_complete.shape[0]

        y_predicted_classes = []
        y_predicted_clusters = self.labels_[self.x_train.shape[0]:]
        assert len(y_predicted_clusters) == X.shape[0]

        # predict top X most frequent intra-cluster tags for each test post
        for y_predicted_cluster in y_predicted_clusters:
            y = np.array([0] * self.y_train.shape[1])  # store inter-cluster occurrence for each tag
            for idx, y_train_data in enumerate(self.y_train):
                if self.labels_[idx] == y_predicted_cluster:  # train and test data are in same cluster
                    y += y_train_data

            # predict most frequent tags (= tags with highest tag occurrences)
            largest_values = heapq.nlargest(self.n_suggested_tags, y)
            n_tag_assignments = 0

            for idx, tag_data in enumerate(y):
                if tag_data in largest_values and n_tag_assignments < self.n_suggested_tags:
                    y[idx] = 1  # this post belongs to this tag
                    n_tag_assignments += 1
                else:
                    y[idx] = 0  # this post does not belong to this tag

            assert sum(y) == self.n_suggested_tags
            y_predicted_classes.append(y)
            
        return np.array(y_predicted_classes)

    def fit(self, X, y=None):
        self.x_train = X
        self.y_train = y
        return self
