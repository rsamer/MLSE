from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from entities.tag import sort_tags_by_frequency


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

    model = KMeans(n_clusters=number_of_clusters, init='k-means++', max_iter=1000, n_init=10, tol=0.00004)
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

    for post in posts_of_cluster:
        for tag in post.tags:
            tag.count = 0

    for post1 in posts_of_cluster:
        for tag1 in post1.tags:
            for post2 in posts_of_cluster:
                for tag2 in post2.tags:
                    if tag1.name == tag2.name:
                        tag1.count += 1

    tags_of_cluster_sorted = sort_tags_by_frequency(reduce(lambda x, y: x+y, [list(post.tags) for post in posts_of_cluster]))

    tag_recommendations = []
    for tag in tags_of_cluster_sorted:
        if tag not in tag_recommendations:
            tag_recommendations.append(tag)

    print "Tags for new post = " + str(tag_recommendations[0:10])
