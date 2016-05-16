# -*- coding: utf-8 -*-

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from entities.tag import Tag
from entities.post import Post


def _posts_for_cluster(model, cluster_number, posts):
    cluster_posts = []

    for idx in range(len(posts) - 1):
        if model.labels_[idx] == cluster_number:
            cluster_posts += [posts[idx]]

    return cluster_posts


def kmeans(number_of_clusters, train_posts, test_posts):

    documents = [" ".join(post.tokens) for post in train_posts + test_posts]

    vectorizer = TfidfVectorizer(stop_words=None)
    X = vectorizer.fit_transform(documents)
    print("n_samples: %d, n_features: %d" % X.shape)

    #     n_jobs = number of jobs to use for computation. This works by computing each of the n_init runs in parallel.
    #     -1 = all CPUs are used
    #      1 = no parallel computing (debugging)
    #      < -1 (n_cpus + 1 + n_jobs) are used. Thus for n_jobs = -2, all CPUs but one are used.
    model = KMeans(n_clusters=number_of_clusters, init='k-means++', max_iter=1000, n_init=10, verbose=True, tol=0.00004, n_jobs=-1)
    model.fit(X)

#     from sklearn.cluster import MiniBatchKMeans
#     model = MiniBatchKMeans(n_clusters=number_of_clusters, init='k-means++', max_iter=1000, n_init=3, verbose=True)  # Take a good look at the docstring and set options here
#     model.fit(X)

    print("Top terms per cluster:")
    order_centroids = model.cluster_centers_.argsort()[:, ::-1]
    terms = vectorizer.get_feature_names()

    for i in range(number_of_clusters):
        print "Cluster %d:" % i,
        for ind in order_centroids[i, :10]:
            print ' %s' % terms[ind],
        print

    test_posts_tag_recommendations = []
    print "="*80
    # TODO: REVIEW THIS!!!
    for i in range(len(test_posts))[::-1]: # reverse order
        test_post_cluster = model.labels_[-(i+1)]
        posts_of_cluster = _posts_for_cluster(model, test_post_cluster, train_posts)
        tags_of_cluster = Post.copied_new_counted_tags_for_posts(posts_of_cluster)
        tags_of_cluster_sorted = Tag.sort_tags_by_frequency(tags_of_cluster)
        print "Tags for new post = " + str(tags_of_cluster_sorted[0:10])
        post = test_posts[len(test_posts) - (i + 1)]
        post.tag_set_prediction = tags_of_cluster_sorted[:1]
        test_posts_tag_recommendations += [tags_of_cluster_sorted[0:10]]
    return test_posts_tag_recommendations
