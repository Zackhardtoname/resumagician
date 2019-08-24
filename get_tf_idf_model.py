#!/usr/bin/env python
# coding: utf-8
import pickle
import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

exclude_percentage = .85
max_vocab = 10000
file_path = "./data/data_scientist_v1.json"

with open(file_path) as json_file:
    data = json.load(json_file)

stopwords = pickle.load(open('./resources/stopwords_set.plk', 'rb'))

with open('./resources/stopwords_set.plk') as json_file:
    data = json.load(json_file)

texts = [item["text"] for item in data]

count_vectorizer = CountVectorizer(max_df=exclude_percentage, stop_words=stopwords, max_features=max_vocab) # Let's limit our vocabulary size to 10,000
word_count_vector=count_vectorizer.fit_transform(texts)

tfidf_transformer = TfidfTransformer(use_idf=False)
tfidf_transformer.fit(word_count_vector)

with open('models/tfidf_transformer.pkl', 'wb') as fp:
    pickle.dump(tfidf_transformer, fp)

with open('models/count_vectorizer.pkl', 'wb') as fp:
    pickle.dump(count_vectorizer, fp)