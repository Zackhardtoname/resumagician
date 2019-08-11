#!/usr/bin/env python
# coding: utf-8

# ## Extracting Important Keywords from Text with TF-IDF and Python's Scikit-Learn 
# 
# Back in 2006, when I had to use TF-IDF for keyword extraction in Java, I ended up writing all of the code from scratch as Data Science nor GitHub were a thing back then and libraries were just limited. The world is much different today. You have several [libraries](http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfTransformer.html#sklearn.feature_extraction.text.TfidfTransformer) and [open-source code on Github](https://github.com/topics/tf-idf?o=desc&s=forks) that provide a decent implementation of TF-IDF. If you don't need a lot of control over how the TF-IDF math is computed then I would highly recommend re-using libraries from known packages such as [Spark's MLLib](https://spark.apache.org/docs/2.2.0/mllib-feature-extraction.html) or [Python's scikit-learn](http://scikit-learn.org/stable/). 
# 
# The one problem that I noticed with these libraries is that they are meant as a pre-step for other tasks like clustering, topic modeling and text classification. [TF-IDF](https://en.wikipedia.org/wiki/Tf%E2%80%93idf) can actually be used to extract important keywords from a document to get a sense of what characterizes a document. For example, if you are dealing with wikipedia articles, you can use tf-idf to extract words that are unique to a given article. These keywords can be used as a very simple summary of the document, it can be used for text-analytics (when we look at these keywords in aggregate), as candidate labels for a document and more. 
# 
# In this article, I will show you how you can use scikit-learn to extract top keywords for a given document using its tf-idf modules. We will specifically do this on a stackoverflow dataset. 

# ## Dataset
# Since we used some pretty clean user reviews in some of my previous tutorials, in  this example, we will be using a Stackoverflow dataset which is slightly noisier and simulates what you could be dealing with in real life. You can find this dataset in [my tutorial repo](https://github.com/kavgan/data-science-tutorials/tree/master/tf-idf/data). Notice that there are two files, the larger file with (20,000 posts)[https://github.com/kavgan/data-science-tutorials/tree/master/tf-idf/data] is used to compute the Inverse Document Frequency (IDF) and the smaller file with [500 posts](https://github.com/kavgan/data-science-tutorials/tree/master/tf-idf/data) would be used as a test set for us to extract keywords from. This dataset is based on the publicly available [Stackoverflow dump on Google's Big Query](https://cloud.google.com/bigquery/public-data/stackoverflow).
# 
# Let's take a peek at our dataset. The code below reads a one per line json string from `data/stackoverflow-data-idf.json` into a pandas data frame and prints out its schema and total number of posts. Here, `lines=True` simply means we are treating each line in the text file as a separate json string. With this, the json in line 1 is not related to the json in line 2.

# In[1]:


import pandas as pd

# read json into a dataframe
df_idf=pd.read_json("data/stackoverflow-data-idf.json",lines=True)

# print schema
print("Schema:\n\n",df_idf.dtypes)
print("Number of questions,columns=",df_idf.shape)


# Take note that this stackoverflow dataset contains 19 fields including post title, body, tags, dates and other metadata which we don't quite need for this tutorial. What we are mostly interested in for this tutorial is the `body` and `title` which is our source of text. We will now create a field that combines both body and title so we have it in one field. We will also print the second `text` entry in our new field just to see what the text looks like.

# In[2]:


import re
def pre_process(text):
    
    # lowercase
    text=text.lower()
    
    #remove tags
    text=re.sub("</?.*?>"," <> ",text)
    
    # remove special characters and digits
    text=re.sub("(\\d|\\W)+"," ",text)
    
    return text

df_idf['text'] = df_idf['title'] + df_idf['body']
df_idf['text'] = df_idf['text'].apply(lambda x:pre_process(x))

#show the first 'text'
df_idf['text'][2]


# Hmm, doesn't look very pretty with all the html in there, but that's the point. Even in such a mess we can extract some great stuff out of this. While you can eliminate all code from the text, we will keep the code sections for this tutorial for the sake of simplicity.  

# ## Creating the IDF
# 
# ### CountVectorizer to create a vocabulary and generate word counts
# The next step is to start the counting process. We can use the CountVectorizer to create a vocabulary from all the text in our `df_idf['text']` and generate counts for each row in `df_idf['text']`. The result of the last two lines is a sparse matrix representation of the counts, meaning each column represents a word in the vocabulary and each row represents the document in our dataset where the values are the word counts. Note that with this representation, counts of some words could be 0 if the word did not appear in the corresponding document.

# In[3]:


from sklearn.feature_extraction.text import CountVectorizer
import re

def get_stop_words(stop_file_path):
    """load stop words """
    
    with open(stop_file_path, 'r', encoding="utf-8") as f:
        stopwords = f.readlines()
        stop_set = set(m.strip() for m in stopwords)
        return frozenset(stop_set)

#load a set of stop words
stopwords=get_stop_words("resources/stopwords.txt")

#get the text column 
docs=df_idf['text'].tolist()

#create a vocabulary of words, 
#ignore words that appear in 85% of documents, 
#eliminate stop words
cv=CountVectorizer(max_df=0.85,stop_words=stopwords)
word_count_vector=cv.fit_transform(docs)


# Now let's check the shape of the resulting vector. Notice that the shape below is `(20000,149391)` because we have 20,000 documents in our dataset (the rows) and the vocabulary size is `149391` meaning we have `149391` unique words (the columns) in our dataset minus the stopwords. In some of the text mining applications, such as clustering and text classification we limit the size of the vocabulary. It's really easy to do this by setting `max_features=vocab_size` when instantiating CountVectorizer.

# In[4]:


word_count_vector.shape


# Let's limit our vocabulary size to 10,000

# In[5]:


cv=CountVectorizer(max_df=0.85,stop_words=stopwords,max_features=10000)
word_count_vector=cv.fit_transform(docs)
word_count_vector.shape


# Now, let's look at 10 words from our vocabulary. Sweet, these are mostly programming related.

# In[6]:


list(cv.vocabulary_.keys())[:10]


# We can also get the vocabulary by using `get_feature_names()`

# In[7]:


list(cv.get_feature_names())[2000:2015]


# ### TfidfTransformer to Compute Inverse Document Frequency (IDF) 
# In the code below, we are essentially taking the sparse matrix from CountVectorizer to generate the IDF when you invoke `fit`. An extremely important point to note here is that the IDF should be based on a large corpora and should be representative of texts you would be using to extract keywords. I've seen several articles on the Web that compute the IDF using a handful of documents. To understand why IDF should be based on a fairly large collection, please read this [page from Standford's IR book](https://nlp.stanford.edu/IR-book/html/htmledition/inverse-document-frequency-1.html).

# In[8]:


from sklearn.feature_extraction.text import TfidfTransformer

tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)
tfidf_transformer.fit(word_count_vector)


# Let's look at some of the IDF values:

# In[9]:


tfidf_transformer.idf_


# ## Computing TF-IDF and Extracting Keywords

# Once we have our IDF computed, we are now ready to compute TF-IDF and extract the top keywords. In this example, we will extract top keywords for the questions in `data/stackoverflow-test.json`. This data file has 500 questions with fields identical to that of `data/stackoverflow-data-idf.json` as we saw above. We will start by reading our test file, extracting the necessary fields (title and body) and get the texts into a list.

# In[10]:


# read test docs into a dataframe and concatenate title and body
df_test=pd.read_json("data/stackoverflow-test.json",lines=True)
df_test['text'] = df_test['title'] + df_test['body']
df_test['text'] =df_test['text'].apply(lambda x:pre_process(x))

# get test docs into a list
docs_test=df_test['text'].tolist()
docs_title=df_test['title'].tolist()
docs_body=df_test['body'].tolist()


# In[11]:


def sort_coo(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)

def extract_topn_from_vector(feature_names, sorted_items, topn=10):
    """get the feature names and tf-idf score of top n items"""
    
    #use only topn items from vector
    sorted_items = sorted_items[:topn]

    score_vals = []
    feature_vals = []

    for idx, score in sorted_items:
        fname = feature_names[idx]
        
        #keep track of feature name and its corresponding score
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])

    #create a tuples of feature,score
    #results = zip(feature_vals,score_vals)
    results= {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]]=score_vals[idx]
    
    return results


# The next step is to compute the tf-idf value for a given document in our test set by invoking `tfidf_transformer.transform(...)`. This generates a vector of tf-idf scores. Next, we sort the words in the vector in descending order of tf-idf values and then iterate over to extract the top-n items with the corresponding feature names, In the example below, we are extracting keywords for the first document in our test set. 
# 
# The `sort_coo(...)` method essentially sorts the values in the vector while preserving the column index. Once you have the column index then its really easy to look-up the corresponding word value as you would see in `extract_topn_from_vector(...)` where we do `feature_vals.append(feature_names[idx])`.

# In[12]:


# you only needs to do this once
feature_names=cv.get_feature_names()

# get the document that we want to extract keywords from
doc=docs_test[0]

#generate tf-idf for the given document
tf_idf_vector=tfidf_transformer.transform(cv.transform([doc]))

#sort the tf-idf vectors by descending order of scores
sorted_items=sort_coo(tf_idf_vector.tocoo())

#extract only the top n; n here is 10
keywords=extract_topn_from_vector(feature_names,sorted_items,10)

# now print the results
print("\n=====Title=====")
print(docs_title[0])
print("\n=====Body=====")
print(docs_body[0])
print("\n===Keywords===")
for k in keywords:
    print(k,keywords[k])


# From the keywords above, the top keywords actually make sense, it talks about `eclipse`, `maven`, `integrate`, `war` and `tomcat` which are all unique to this specific question. There are a couple of kewyords that could have been eliminated such as `possibility` and perhaps even `project` and you can do this by adding more common words to your stop list and you can even create your own set of stop list, very specific to your domain as [described here](http://kavita-ganesan.com/tips-for-constructing-custom-stop-word-lists/).
# 
# 

# In[13]:


# put the common code into several methods
def get_keywords(idx):

    #generate tf-idf for the given document
    tf_idf_vector=tfidf_transformer.transform(cv.transform([docs_test[idx]]))

    #sort the tf-idf vectors by descending order of scores
    sorted_items=sort_coo(tf_idf_vector.tocoo())

    #extract only the top n; n here is 10
    keywords=extract_topn_from_vector(feature_names,sorted_items,10)
    
    return keywords

def print_results(idx,keywords):
    # now print the results
    print("\n=====Title=====")
    print(docs_title[idx])
    print("\n=====Body=====")
    print(docs_body[idx])
    print("\n===Keywords===")
    for k in keywords:
        print(k,keywords[k])


# Now let's look at keywords generated for a much longer question: 
# 

# In[14]:


idx=120
keywords=get_keywords(idx)
print_results(idx,keywords)


# Whoala! Now you can extract important keywords from any type of text! 
