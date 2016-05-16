# -*- coding: utf-8 -*-

from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
from entities.tag import Tag
from entities.post import Post


def _posts_for_cluster(model, cluster_number, posts):
    cluster_posts = []

    for idx in range(len(posts) - 1):
        if model.labels_[idx] == cluster_number:
            cluster_posts += [posts[idx]]

    return cluster_posts


def hac(number_of_clusters, posts, new_posts):

    documents = [" ".join(post.tokens) for post in posts + new_posts]

    vectorizer = TfidfVectorizer(stop_words=None)
    X = vectorizer.fit_transform(documents)

    #C = 1 - cosine_similarity(X.T)
    model = AgglomerativeClustering(n_clusters=number_of_clusters, linkage='ward', affinity='euclidean')
    model.fit(X.toarray())

    posts_tag_recommendations = []
    for i in range(len(new_posts))[::-1]:  # reverse order
        new_post_cluster = model.labels_[-(i + 1)]
        posts_of_cluster = _posts_for_cluster(model, new_post_cluster, posts)
        tags_of_cluster = Post.copied_new_counted_tags_for_posts(posts_of_cluster)
        tags_of_cluster_sorted = Tag.sort_tags_by_frequency(tags_of_cluster)
        print "Tags for new post = " + str(tags_of_cluster_sorted[0:10])
        posts_tag_recommendations += [tags_of_cluster_sorted[0:10]]

        post = new_posts[len(new_posts) - (i + 1)]
        post.tag_set_prediction = tags_of_cluster_sorted[0:2]
    return posts_tag_recommendations
