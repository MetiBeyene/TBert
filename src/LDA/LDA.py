# Importing modules
import pandas as pd
# from src.loaders.load_data import load_data
import os
import re


# # Read data into papers
papers = pd.read_table('../../data/MSRP/MSRParaphraseCorpus/msr-para-val.tsv', sep='\t',quoting = 3)
# print(papers.head(5))
# print(papers['#1 String'])


# # Remove the columns
papers = papers.drop(columns=['Quality', '#1 ID', '#2 ID'], axis=1).sample(100)
print(list(papers.columns))
# # Print out the first rows of papers
# print(papers['#1 String'])
#
dict = {'#1 String': 'String1',
        '#2 String': 'String2'}
papers.rename(columns=dict, inplace=True)
# print(list(papers.columns))
# print(papers['String1'])


papers['paper_String1_processed'] = \
papers['String1'].map(lambda x: re.sub('[,\.!?]', '', x))
# Convert the titles to lowercase
papers['paper_String1_processed'] = \
papers['paper_String1_processed'].map(lambda x: x.lower())
# Print out the first rows of papers
print(papers['paper_String1_processed'].head())


papers['paper_String2_processed'] = \
papers['String2'].map(lambda x: re.sub('[,\.!?]', '', x))
# Convert the titles to lowercase
papers['paper_String2_processed'] = \
papers['paper_String2_processed'].map(lambda x: x.lower())
# Print out the first rows of papers
print(papers['paper_String2_processed'].head())

import gensim
from gensim.utils import simple_preprocess
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use'])

def sent_to_words(sentences):
    for sentence in sentences:
        # deacc=True removes punctuations
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))

def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc))
             if word not in stop_words] for doc in texts]


data = papers.paper_String1_processed.values.tolist()
data_words = list(sent_to_words(data))

# remove stop words
data_words = remove_stopwords(data_words)

# print(data_words[:1][0][:30])

import gensim.corpora as corpora
#
# # Create Dictionary
id2word = corpora.Dictionary(data_words)
# print(id2word)
#
# # Create Corpus
texts = data_words
# print(texts)
#
# # Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]
# print(corpus)
#
# # View
print(corpus[:1][0][:30])
#
from pprint import pprint
#
# # number of topics
num_topics = 10
#
# # Build LDA model

lda_model = gensim.models.LdaModel(corpus=corpus,id2word=id2word,num_topics=num_topics)
print(lda_model)

# # Print the Keyword in the 10 topics
pprint(lda_model.print_topics())
doc_lda = lda_model[corpus]
import pyLDAvis.gensim
import pickle
import pyLDAvis

# # Visualize the topics
# pyLDAvis.enable_notebook()
#
LDAvis_data_filepath = os.path.join(str(num_topics))

# # this is a bit time consuming - make the if statement True
# # if you want to execute visualization prep yourself
if 1 == 1:
    LDAvis_prepared = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
    with open(LDAvis_data_filepath, 'wb') as f:
        pickle.dump(LDAvis_prepared, f)

# load the pre-prepared pyLDAvis data from disk
with open(LDAvis_data_filepath, 'rb') as f:
    LDAvis_prepared = pickle.load(f)

pyLDAvis.save_html(LDAvis_prepared, str(num_topics) +'.html')

LDAvis_prepared