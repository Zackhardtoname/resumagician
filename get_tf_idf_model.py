#!/usr/bin/env python
# coding: utf-8
import pickle
import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

exclude_percentage = .05
max_vocab = 20000
file_path = "data/software_engineer.json"

with open(file_path) as json_file:
    data = json.load(json_file)

stopwords = pickle.load(open('./resources/stopwords_set.pkl', 'rb'))

texts = [item["text"] for item in data.values()]

count_vectorizer = CountVectorizer(max_df=exclude_percentage, stop_words=stopwords, max_features=max_vocab) # Let's limit our vocabulary size to 10,000
word_count_vector=count_vectorizer.fit_transform(texts)

tfidf_transformer = TfidfTransformer(use_idf=False)
tfidf_transformer.fit(word_count_vector)

with open('models/tfidf_transformer.pkl', 'wb') as fp:
    pickle.dump(tfidf_transformer, fp, pickle.HIGHEST_PROTOCOL)

with open('models/count_vectorizer.pkl', 'wb') as fp:
    pickle.dump(count_vectorizer, fp, pickle.HIGHEST_PROTOCOL)