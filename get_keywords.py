import helpers
import pickle

with open('tfidf_transformer.obj', 'rb') as fp:
    tfidf_transformer = pickle.load(fp)
with open('count_vectorizer.obj', 'rb') as fp:
    count_vectorizer = pickle.load(fp)
feature_names = count_vectorizer.get_feature_names()

doc = """"""
#generate tf-idf for the given document
tf_idf_vector=tfidf_transformer.transform(count_vectorizer.transform([doc]))

#sort the tf-idf vectors by descending order of scores
sorted_items = helpers.sort_coo(tf_idf_vector.tocoo())
#extract only the top n; n here is 10
keywords = helpers.extract_topn_from_vector(feature_names,sorted_items, 10)