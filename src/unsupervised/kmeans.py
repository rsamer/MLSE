# -*- coding: utf-8 -*-

import heapq
import numpy as np
from sklearn.cluster import KMeans


class CustomKMeans(KMeans):
    y_train = None

    def predict(self, X, n_suggested_tags=2):
        y_predicted_clusters = super(CustomKMeans, self).predict(X)
        y_predicted_classes = []

        for y_predicted_cluster in y_predicted_clusters:
            y = [0]
            for idx, label in enumerate(self.labels_):
                if label == y_predicted_cluster:
                    y += self.y_train[idx]
            largest_values = heapq.nlargest(n_suggested_tags, y)
            n_tag_assignments = 0

            for idx, tag_assignments in enumerate(y):
                if tag_assignments > 1 and tag_assignments in largest_values and n_tag_assignments < n_suggested_tags:
                    y[idx] = 1
                    n_tag_assignments += 1
                else:
                    y[idx] = 0

            #assert sum(y) == n_tags
            y_predicted_classes.append(y.tolist())
        y_predicted_classes = np.array(y_predicted_classes)
        return y_predicted_classes

    def fit(self, X, y=None):
        self.y_train = y
        return super(CustomKMeans, self).fit(X, y)
