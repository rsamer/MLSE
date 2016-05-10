from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from entities.tag import sort_tags_by_frequency


def _posts_for_cluster(model, cluster_number, posts):
    cluster_posts = []

    for idx in range(len(posts) - 1):
        if model.labels_[idx] == cluster_number:
            cluster_posts += [posts[idx]]

    return cluster_posts


def kmeans(tags, posts, new_post):
    true_k = len(posts) / 10

    documents = [" ".join(post.body_tokens) for post in posts + [new_post]]

    vectorizer = TfidfVectorizer(stop_words=None)
    X = vectorizer.fit_transform(documents)

    model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
    model.fit(X)

    print("Top terms per cluster:")
    order_centroids = model.cluster_centers_.argsort()[:, ::-1]
    terms = vectorizer.get_feature_names()

    for i in range(true_k):
        print "Cluster %d:" % i,
        for ind in order_centroids[i, :10]:
            print ' %s' % terms[ind],
        print

    new_post_cluster = model.labels_[-1]
    posts_of_cluster = _posts_for_cluster(model, new_post_cluster, posts)
    tags_of_cluster_sorted = sort_tags_by_frequency(reduce(lambda x, y: x+y, [list(post.tags) for post in posts_of_cluster]))

    print "Tags for new post ="
    for tag in tags_of_cluster_sorted[:50]:
        print tag
