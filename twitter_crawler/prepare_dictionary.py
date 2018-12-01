
from __future__ import print_function

import csv

import langid
import logging
import nltk
import numpy as np
import re
from collections import defaultdict
from gensim import corpora
from optparse import OptionParser
from string import digits


# Get the documents from the DB
from twitter_crawler.constants import PRESIDENTS


# --------------------------------------
#  Clean documents functions
# --------------------------------------

def remove_urls(text):
    text = re.sub(r"(?:\@|http?\://)\S+", "", text)
    text = re.sub(r"(?:\@|https?\://)\S+", "", text)
    return text




def stop_words_list():
    '''
        A stop list specific to the observed timelines composed of noisy words
        This list would change for different set of timelines
    '''
    return ['amp', 'get', 'got', 'hey', 'hmm', 'hoo', 'hop', 'iep', 'let', 'ooo', 'par',
            'pdt', 'pln', 'pst', 'wha', 'yep', 'yer', 'aest', 'didn', 'nzdt', 'via',
            'one', 'com', 'new', 'like', 'great', 'make', 'top', 'awesome', 'best',
            'good', 'wow', 'yes', 'say', 'yay', 'would', 'thanks', 'thank', 'going',
            'new', 'use', 'should', 'could', 'really', 'see', 'want', 'nice',
            'while', 'know', 'free', 'today', 'day', 'always', 'last', 'put', 'live',
            'week', 'went', 'wasn', 'was', 'used', 'ugh', 'try', 'kind', 'http', 'much',
            'need', 'next', 'app', 'ibm', 'appleevent', 'using']


def all_stopwords(tokenized_documents):
    '''
        Builds a stoplist composed of stopwords in several languages,
        tokens with one or 2 words and a manually created stoplist
    '''
    # tokens with 1 characters
    unigrams = [w for w in tokenized_documents if len(w) == 1]
    # tokens with 2 characters
    bigrams = [w for w in tokenized_documents if len(w) == 2]

    # Compile global list of stopwords
    stoplist = set(nltk.corpus.stopwords.words("english")
                   + nltk.corpus.stopwords.words("french")
                   + nltk.corpus.stopwords.words("spanish")
                   + stop_words_list()
                   + unigrams + bigrams)
    return stoplist



def count_token(tweets):
    '''
        Calculates the number of occurence of each word across the whole corpus
    '''
    token_frequency = defaultdict(int)
    for doc in tweets:
        for token in doc['tokens']:
            token_frequency[token] += 1
    return token_frequency


def token_condition(token, token_frequency, stoplist):
    '''
        Only keep a token that is not in the stoplist,
        and with frequency > 1 among all documents
    '''
    return (token not in stoplist and len(token.strip(digits)) == len(token)
            and token_frequency[token] > 1)


def keep_best_tokens(tweets_collection, tokens_frecuency, stop_words):
    '''
        Removes all tokens that do not satistify a certain condition
    '''
    tweets_collection_new = []
    for tweet in tweets_collection:
        tokens = []
        for token in tweet['tokens']:
            if token_condition(token , tokens_frecuency, stop_words):
                tokens.append(token)
        tweet['tokens'] = tokens
        tweets_collection_new.append(tweet)
    return tweets_collection_new

# ---------------------------------------------------------
#  Main
# ---------------------------------------------------------

print(__doc__)
# Display progress logs on stdout
logging.basicConfig(level=logging.INFO, format='>>> %(asctime)s %(levelname)s %(message)s')

# ---------------------------------------------------------
#  parse commandline arguments
# ---------------------------------------------------------

op = OptionParser()


def read_tweet(president):
    tweet_list = []
    line_count = 0
    with open('../data/presidents/%s_%s_tweets.csv' % (president['account'], president['country']), 'rt') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')

        for row in csv_reader:
            tweet = {}
            tweet['country'] = row[0]
            tweet['id'] = row[1]
            tweet['date'] = row[2]
            tweet['text'] = row[3]
            tweet_list.append(tweet)
            line_count = line_count + 1
    print('President -- ' + president['country'] + " -- Tweets -- " + str(line_count))
    return tweet_list


def filter_by_length_and_remove_urls(tweets, percent=25):
    tweet_collection_new = []
    for tweet_collection in tweets:
        for tweet in tweet_collection:
            if len(tweet['text']) > percent:
                tweet_new = {}
                tweet_new['country'] = tweet['country']
                tweet_new['id'] = tweet['id']
                tweet_new['date'] = tweet['date']
                tweet_new['text'] = remove_urls(tweet['text'])
                tweet_collection_new.append(tweet_new)
        else:
            print('Tweet Removed -- ' + tweet['text'])

    return tweet_collection_new

# This returns a list of tokens / single words for each user
def tokenize_doc(tweets):
    '''
        Tokenizes the raw text of each document
    '''
    tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
    tweet_collection_new = []
    for tweet in tweets:
        tweet_new = {}
        tweet_new['country'] = tweet['country']
        tweet_new['id'] = tweet['id']
        tweet_new['date'] = tweet['date']
        tweet_new['text'] = remove_urls(tweet['text'])
        tweet_new['tokens'] = tokenizer.tokenize(tweet_new['text'].lower())
        tweet_collection_new.append(tweet_new)
        print('Number tokens -- ' + str(len(tweet_new['tokens'])))

    return tweet_collection_new


# --------------- Main --------------- #

president_tweets = []
for president in PRESIDENTS: 
    tweets = read_tweet(president)
    president_tweets.append(tweets)

# ---------------------------------------------------------
#  Documents / timelines selection and clean up
# ---------------------------------------------------------

# Keep 1st Quartile of documents by length and filter out non-English words
tweets_collection = filter_by_length_and_remove_urls(president_tweets, 25)

print("\nWe have " + str(len(tweets_collection)) + " documents in english ")
print()

# ---------------------------------------------------------
#  Tokenize documents
# ---------------------------------------------------------

# At this point tokenized_documents.keys() == ['user_id', 'tokens']
tweets_collection = tokenize_doc(tweets_collection)

token_frequency = count_token(tweets_collection)
stoplist = all_stopwords(token_frequency)
tweets_collection = keep_best_tokens(tweets_collection, token_frequency, stoplist)

# for visualization purposes only
for doc in tweets_collection:
    doc['tokens'].sort()

# ---------------------------------------------------------
#  Save tokenized docs in database
# ---------------------------------------------------------
# We save the tokenized version of the raw text in the db

# ---------------------------------------------------------
#  Dictionary and Corpus
# ---------------------------------------------------------

# build the dictionary
dictionary = corpora.Dictionary([doc['tokens'] for doc in tweets_collection])
dictionary.compactify()

# We now have a dictionary with N unique tokens
print("Dictionary: ", end=' ')
print(dictionary)
print()

# and save the dictionary for future use
dictionary.save("../data/presidents/all-presidents-dictionary.dic")

# Build the corpus: vectors with occurence of each word for each document
# and convert tokenized documents to vectors
corpus = [dictionary.doc2bow(doc['tokens']) for doc in tweets_collection]

# and save in Market Matrix format
corpora.MmCorpus.serialize("../data/presidents/all-presidents-corpus.mm", corpus)
