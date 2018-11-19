import numpy as np
from retpred.tweets import extract_content, extract_metadata, extract_labels
from retpred.utils.io import load_tweets, save_txt, save_array

# define constants
FILES = ['tweets_1.json', 
    'tweets_2.json', 
    'tweets_3.json', 
    'tweets_4.json', 
    'tweets_5.json']
PATH = '/storage/'

# load tweets from file
tweets = []
for f in FILES:
    fname = PATH + f
    t = load_tweets(fname)
    print('Loaded {} tweets from file {}'.format(len(t), fname))
    tweets += t
print('Loaded {} tweets for further processing'.format(len(tweets)))

# extract tweet content
texts = extract_content(tweets)
fname = PATH + 'tweet_texts.txt'
save_txt(fname, texts)
print('Saved tokenized tweet texts to file {}'.format(fname))

# extract metadata
meta = extract_metadata(tweets)
fname = PATH + 'tweet_metadata.hdf5'
save_array(fname, meta, 'tweet_metadata')
print('Saved {} features for each tweet to file {}'.format(meta.shape[1], fname))

# extract labels
labels = extract_labels(tweets)
fname = PATH + 'tweet_labels.hdf5'
save_array(fname, labels, 'tweet_labels')
print('Saved raw labels to file {}'.format(fname))
labels = np.log1p(labels)
fname = PATH + 'tweet_log_labels.hdf5'
save_array(fname, labels, 'tweet_log_labels')
print('Saved log1p-transformed labels to file {}'.format(fname))
