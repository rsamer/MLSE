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


def kmeans(number_of_clusters, posts, new_post):

    documents = [" ".join(post.tokens) for post in posts + [new_post]]

    vectorizer = TfidfVectorizer(stop_words=None)
    X = vectorizer.fit_transform(documents)

    #     n_jobs = number of jobs to use for computation. This works by computing each of the n_init runs in parallel.
    #     -1 = all CPUs are used
    #      1 = no parallel computing (debugging)
    #      < -1 (n_cpus + 1 + n_jobs) are used. Thus for n_jobs = -2, all CPUs but one are used.
    model = KMeans(n_clusters=number_of_clusters, init='k-means++', max_iter=1000, n_init=10, tol=0.00004, n_jobs=-1)
    model.fit(X)

    print("Top terms per cluster:")
    order_centroids = model.cluster_centers_.argsort()[:, ::-1]
    terms = vectorizer.get_feature_names()

    for i in range(number_of_clusters):
        print "Cluster %d:" % i,
        for ind in order_centroids[i, :10]:
            print ' %s' % terms[ind],
        print

    new_post_cluster = model.labels_[-1]
    posts_of_cluster = _posts_for_cluster(model, new_post_cluster, posts)

    Post.update_tag_counts_according_to_given_post_list(posts_of_cluster)

    tags_of_cluster_sorted = Tag.sort_tags_by_frequency(reduce(lambda x, y: x+y, [list(post.tag_set) for post in posts_of_cluster]))

    tag_recommendations = []
    for tag in tags_of_cluster_sorted:
        if tag not in tag_recommendations:
            tag_recommendations.append(tag)

    print "Tags for new post = " + str(tag_recommendations[0:10])
    return tag_recommendations
