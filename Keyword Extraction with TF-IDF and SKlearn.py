#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import pickle
import helpers

df_idf = pd.read_json("data/stackoverflow-data-idf.json", lines=True)

df_idf['text'] = df_idf['title'] + df_idf['body']
df_idf['text'] = df_idf['text'].apply(lambda x: helpers.pre_process(x))

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
stopwords = pickle.load(open('./resources/stopwords_set.plk', 'rb'))

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

tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=False)
tfidf_transformer.fit(word_count_vector)


# Let's look at some of the IDF values:

# In[9]:


# tfidf_transformer.idf_


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
# doc=docs_test[0]
doc = """
7/30/2019 Job Application for Software Engineer, Data (Intern/Co-op) at Robinhood
https://boards.greenhouse.io/robinhood/jobs/1739582 1/5
Software Engineer, Data (Intern/Co-op)
at Robinhood (View all jobs)
Menlo Park, CA
About the company
Robinhood is democratizing our financial system. We offer commission-free investing in stocks, ETFs,
options, and cryptocurrencies. Robinhood Financial, our broker-dealer, is a fast-growing brokerage with
millions of users and billions of dollars in transaction volume. Robinhood has received the Apple Design
Award (2015), the Google Play Award for Best Use of Material Design (2016), and was named Fast
Company’s 11th Most Innovative Company in the world (2016). We’re backed with $539 million in capital
from top-tier investors such as DST Global, NEA, Index Ventures, Thrive Capital, Sequoia, and KPCB,
and were most recently valued at $5.6 billion. Robinhood is based in Menlo Park, California, and Lake
Mary, Florida.
About the role
Software engineers on the Data team are the central node to every team within Robinhood. You’ll be
interfacing with UX researchers to gain insight, Ops engineers to ensure successful deployments, and
Data Scientists to advance our product by utilizing machine intelligence. The Data team consists of
functions including Data Platform (e.g. our compute infrastructure), Data Products (e.g. newsfeed),
Growth and Marketing (e.g. optimizing top of funnel growth), and Risk and Fraud (e.g. real-time risk
mitigation systems).
Every decision made at Robinhood is backed by data; our company trajectory is defined by the systems,
tools, and analytics powered by our exceptional team. We integrate multifaceted data streams such as
rapidly changing market data, user data based on app activity, and brokerage operations data to perfect
our processes and workflows.
In this role you will:
Have a dedicated mentor who will review your code and help you get ramped up.
Productionize machine learning algorithms, augment data scientists to make the Robinhood
products smarter.
Build scalable APIs in a microservice ecosystem.
Build robust distributed systems to process large scale data streams into useful applications.
Engineer fast, real-time data pipelines to process data from across the financial markets and our
internal event streams.
Implement risk monitoring systems to improve tracking and mitigation of several risks including
money laundering, financial fraud and margin exposure.
Work closely with data scientists and collaborate with back-end and front-end engineers to build
products for gathering timely insights about customer behavior, etc.
Some things we consider critical for this role:
Strong CS fundamentals
Desire to own product development from end-to-end, including research, design, and
implementation.
Comfortable and excited to work with distributed systems and data at scale.
Some things that would be amazing to have for this role:
Previous software engineering experience from internships, hackathons, or side projects.
Familiarity with Python, Golang, data systems.
Technologies we use:
Airflow
Celery (written by our very own Ask Solem)
Faust
Elasticsearch, Logstash, Kibana (ELK)
Kafka
Apply Now
7/30/2019 Job Application for Software Engineer, Data (Intern/Co-op) at Robinhood
https://boards.greenhouse.io/robinhood/jobs/1739582 2/5
Apply for this Job * Required
Redis
Presto
Spark
Hive
Redshift and AWS Suite
Zookeeper
Consul
Note to Recruiters and Placement Agencies: Robinhood does not accept unsolicited agency
resumes. Robinhood does not pay placement fees for candidates submitted by any agency other than
its approved partners.
First Name *
Last Name *
Email *
Phone *
Resume/CV *
Cover Letter
Your full LinkedIn profile
will be shared. Learn More
Apply with LinkedIn
7/30/2019 Job Application for Software Engineer, Data (Intern/Co-op) at Robinhood
https://boards.greenhouse.io/robinhood/jobs/1739582 3/5
LinkedIn Profile *
Website
How did you hear about this job?
Work Authorization *
Please select
Have you used Robinhood? *
--
U.S. Equal Opportunity Employment Information (Completion is voluntary)
Individuals seeking employment at Robinhood are considered without regards to race, color,
religion, national origin, age, sex, marital status, ancestry, physical or mental disability,
veteran status, gender identity, or sexual orientation. You are being given the opportunity to
provide the following information in order to help us comply with federal and state Equal
Employment Opportunity/Affirmative Action record keeping, reporting, and other legal
requirements.
Completion of the form is entirely voluntary. Whatever your decision, it will not be considered
in the hiring process or thereafter. Any information that you do provide will be recorded and
maintained in a confidential file.
Gender
Please select
Are you Hispanic/Latino?
7/30/2019 Job Application for Software Engineer, Data (Intern/Co-op) at Robinhood
https://boards.greenhouse.io/robinhood/jobs/1739582 4/5
Please select
Race & Ethnicity Definitions
If you believe you belong to any of the categories of protected veterans listed below, please
indicate by making the appropriate selection. As a government contractor subject to Vietnam
Era Veterans Readjustment Assistance Act (VEVRAA), we request this information in order to
measure the effectiveness of the outreach and positive recruitment efforts we undertake
pursuant to VEVRAA. Classification of protected categories is as follows:
A "disabled veteran" is one of the following: a veteran of the U.S. military, ground, naval or air
service who is entitled to compensation (or who but for the receipt of military retired pay would
be entitled to compensation) under laws administered by the Secretary of Veterans Affairs; or
a person who was discharged or released from active duty because of a service-connected
disability.
A "recently separated veteran" means any veteran during the three-year period beginning on
the date of such veteran's discharge or release from active duty in the U.S. military, ground,
naval, or air service.
An "active duty wartime or campaign badge veteran" means a veteran who served on active
duty in the U.S. military, ground, naval or air service during a war, or in a campaign or
expedition for which a campaign badge has been authorized under the laws administered by
the Department of Defense.
An "Armed forces service medal veteran" means a veteran who, while serving on active duty
in the U.S. military, ground, naval or air service, participated in a United States military
operation for which an Armed Forces service medal was awarded pursuant to Executive
Order 12985.
Veteran Status
Please select
Form CC-305
OMB Control Number 1250-0005
Expires 1/31/2020
Voluntary Self-Identification of Disability
Why are you being asked to complete this form?
Because we do business with the government, we must reach out to, hire, and provide equal
opportunity to qualified people with disabilities1. To help us measure how well we are doing,
we are asking you to tell us if you have a disability or if you ever had a disability. Completing
this form is voluntary, but we hope that you will choose to fill it out. If you are applying for a
job, any answer you give will be kept private and will not be used against you in any way.
If you already work for us, your answer will not be used against you in any way. Because a
person may become disabled at any time, we are required to ask all of our employees to
update their information every five years. You may voluntarily self-identify as having a
disability on this form without fear of any punishment because you did not identify as having a
disability earlier.
How do I know if I have a disability?
You are considered to have a disability if you have a physical or mental impairment or medical
condition that substantially limits a major life activity, or if you have a history or record of such
an impairment or medical condition.
Disabilities include, but are not limited to:
7/30/2019 Job Application for Software Engineer, Data (Intern/Co-op) at Robinhood
https://boards.greenhouse.io/robinhood/jobs/1739582 5/5
Blindness
Deafness
Cancer
Diabetes
Epilepsy
Autism
Cerebral palsy
HIV/AIDS
Schizophrenia
Muscular dystrophy
Bipolar disorder
Major depression
Multiple sclerosis (MS)
Missing limbs or partially missing limbs
Post-traumatic stress disorder (PTSD)
Obsessive compulsive disorder
Impairments requiring the use of a wheelchair
Intellectual disability (previously called mental retardation)
Disability Status
Please select
Reasonable Accommodation Notice
Federal law requires employers to provide reasonable accommodation to qualified individuals
with disabilities. Please tell us if you require a reasonable accommodation to apply for a job or
to perform your job. Examples of reasonable accommodation include making a change to the
application process or work procedures, providing documents in an alternate format, using a
sign language interpreter, or using specialized equipment.
1Section 503 of the Rehabilitation Act of 1973, as amended. For more information about this
form or the equal employment obligations of Federal contractors, visit the U.S. Department of
Labor's Office of Federal Contract Compliance Programs (OFCCP) website at
www.dol.gov/ofccp.
PUBLIC BURDEN STATEMENT: According to the Paperwork Reduction Act of 1995 no
persons are required to respond to a collection of information unless such collection displays
a valid OMB control number. This survey should take about 5 minutes to complete.
Suubbmiitt Apppplliiccaattiioonn
Powered by
Read our Privacy Policy
"""

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


# # From the keywords above, the top keywords actually make sense, it talks about `eclipse`, `maven`, `integrate`, `war` and `tomcat` which are all unique to this specific question. There are a couple of kewyords that could have been eliminated such as `possibility` and perhaps even `project` and you can do this by adding more common words to your stop list and you can even create your own set of stop list, very specific to your domain as [described here](http://kavita-ganesan.com/tips-for-constructing-custom-stop-word-lists/).
# #
# #
#
# # In[13]:
#
#
# # put the common code into several methods
# def get_keywords(idx):
#
#     #generate tf-idf for the given document
#     tf_idf_vector=tfidf_transformer.transform(cv.transform([docs_test[idx]]))
#
#     #sort the tf-idf vectors by descending order of scores
#     sorted_items=sort_coo(tf_idf_vector.tocoo())
#
#     #extract only the top n; n here is 10
#     keywords=extract_topn_from_vector(feature_names,sorted_items,10)
#
#     return keywords
#
# def print_results(idx,keywords):
#     # now print the results
#     print("\n=====Title=====")
#     print(docs_title[idx])
#     print("\n=====Body=====")
#     print(docs_body[idx])
#     print("\n===Keywords===")
#     for k in keywords:
#         print(k,keywords[k])
#
#
# # Now let's look at keywords generated for a much longer question:
# #
#
# # In[14]:
#
#
# idx=120
# keywords=get_keywords(idx)
# print_results(idx,keywords)
#
#
# # Whoala! Now you can extract important keywords from any type of text!
