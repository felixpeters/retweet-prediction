import h5py
import json
from twitter import Status, Url, User, Hashtag

def save_array(fname, data, dname):
    with h5py.File(fname, 'w') as hf:
        hf.create_dataset(dname, data=data)
    return

def load_array(fname, dname):
    data = []
    with h5py.File(fname, 'r') as hf:
        data = hf[dname][:]
    return data

def save_json(fname, data):
    with open(fname, 'w') as f:
        json.dump(data, f, indent=2, sort_keys=True)
    return

def load_json(fname):
    data = {}
    with open(fname, 'r') as f:
        data = json.load(f)
    return data

def save_txt(fname, data):
    with open(fname, 'w') as f:
        for s in data:
            f.write(s.encode('unicode_escape').decode())
            f.write("\n")
    return

def load_txt(fname):
    data = []
    with open(fname, 'r') as f:
        for line in f:
            data.append(line)
    return data

def load_tweets(fname):
    data = []
    with open(fname) as f:
        for line in f:
            data.append(json.loads(line))
    tweets = [new_tweet_from_json(d) for d in data]
    tweets = [t for t in tweets if t.retweeted_status == None]
    return tweets

def new_tweet_from_json(data):
    tweet = Status.NewFromJsonDict(data)
    if 'urls' in data:
        tweet.urls = [Url.NewFromJsonDict(u) for u in data['urls']]
    else:
        tweet.urls = []
    if 'user_mentions' in data:
        tweet.user_mentions = [User.NewFromJsonDict(u) for u in data['user_mentions']]
    else:
        tweet.user_mentions = []
    if 'hashtags' in data:
        tweet.hashtags = [Hashtag.NewFromJsonDict(h) for h in data['hashtags']]
    else:
        tweet.hashtags = []
    return tweet

