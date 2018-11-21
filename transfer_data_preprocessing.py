import pandas as pd
import numpy as np
from retpred.utils.io import save_array, save_txt, load_json
from retpred.text import tokenize

# define constants
ROOT_PATH = '/storage/'
AFFECTS = ['anger', 'fear', 'joy', 'sadness']

# preprocess irony detection dataset
path = ROOT_PATH + 'irony/semeval18_3/SemEval2018-T3-train-taskB_emoji.txt'
df = pd.read_table(path)

labels = np.array(df['Label'])
save_array(ROOT_PATH + 'irony/irony_labels.hdf5', labels, 'irony_labels')

texts = df['Tweet text']
texts = [tokenize(t) for t in texts]
save_txt(ROOT_PATH + 'irony/irony_texts.txt', texts)

# preprocess topic classification dataset
path = ROOT_PATH + 'topic/us-topics-181109/train.json'
topic_data = load_json(fname=path)

texts = []
labels = []
for tid in topic_data:
    topic = topic_data[tid]
    texts.append(topic['text'])
    labels.append(topic['topic'])

texts = [tokenize(t) for t in texts]
save_txt(ROOT_PATH + 'topic/topic_texts.txt', texts)

labels = pd.Series(data=labels).astype('category')
categories = labels.cat.categories.tolist()
save_txt(ROOT_PATH + 'topic/categories.txt', categories)

labels = np.array(labels.cat.codes)
save_array(ROOT_PATH + 'topic/topic_labels.hdf5', labels, 'topic_labels')

# preprocess offensive language detection dataset
df = pd.read_csv(ROOT_PATH + 'offensive_lang/davison_2017/labeled_data.csv')

categories = ['hate_speech', 'offensive_language', 'neither']
save_txt(ROOT_PATH + 'offensive_lang/categories.txt', categories)

texts = df['tweet']
texts = [tokenize(t) for t in texts]
save_txt(ROOT_PATH + 'offensive_lang/offlang_texts.txt', texts)

labels = np.array(df['class'])
save_array(ROOT_PATH + 'offensive_lang/offlang_labels.hdf5', labels, 'offlang_labels')

# preprocess sentiment analysis datasets
# valence classification dataset
df_voc = pd.read_table(ROOT_PATH + 'sentiment/semeval18_1/2018-Valence-oc-En-train.txt')

labels = df_voc['Intensity Class'].astype('category')
labels = labels.cat.reorder_categories([
    '-3: very negative emotional state can be inferred',
    '-2: moderately negative emotional state can be inferred',
    '-1: slightly negative emotional state can be inferred',
    '0: neutral or mixed emotional state can be inferred',
    '1: slightly positive emotional state can be inferred',
    '2: moderately positive emotional state can be inferred',
    '3: very positive emotional state can be inferred'])

categories = labels.cat.categories.tolist()
save_txt(ROOT_PATH + 'sentiment/valence_categories.txt', categories)

labels = np.array(labels.cat.codes)
save_array(ROOT_PATH + 'sentiment/valence_labels.hdf5', labels, 'valence_labels')

texts = df_voc['Tweet']
texts = [tokenize(t) for t in texts]
save_txt(ROOT_PATH + 'sentiment/valence_texts.txt', texts)

# affect classification datasets
for a in AFFECTS:
    print('processing {} dataset'.format(a))
    path = ROOT_PATH + 'sentiment/semeval18_1/EI-oc-En-{}-train.txt'.format(a)
    df = pd.read_table(path)
    texts = df['Tweet']
    texts = [tokenize(t) for t in texts]
    save_txt(ROOT_PATH + 'sentiment/{}_texts.txt'.format(a), texts)
    labels = np.array(df['Intensity Class'].astype('category').cat.codes)
    save_array(ROOT_PATH + 'sentiment/{}_labels.hdf5'.format(a), labels, '{}_labels'.format(a))
