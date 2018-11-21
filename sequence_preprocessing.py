from retpred.embedding import SequencePreprocessor
from retpred.utils.io import load_txt, save_txt, save_array, save_json

# define constants
PATH = '/storage/'
MAX_SEQ_LEN = 32
NUM_WORDS = 20000
EMB_PATH = PATH + 'glove/glove.twitter.27B.200d.txt'

# load datasets
tweets = load_txt(PATH + 'tweet_texts.txt')
irony = load_txt(PATH + 'irony/irony_texts.txt')
offlang = load_txt(PATH + 'offensive_lang/offlang_texts.txt')
anger = load_txt(PATH + 'sentiment/anger_texts.txt')
joy = load_txt(PATH + 'sentiment/joy_texts.txt')
fear = load_txt(PATH + 'sentiment/fear_texts.txt')
sadness = load_txt(PATH + 'sentiment/sadness_texts.txt')
valence = load_txt(PATH + 'sentiment/valence_texts.txt')
topic = load_txt(PATH + 'topic/topic_texts.txt')
# combine texts into one data set
texts = tweets + irony + offlang + anger + joy + fear + sadness + valence + topic
print('loaded {} tweet texts for sequence preprocessing'.format(len(texts)))

# create sequence preprocessor
sp = SequencePreprocessor(num_words=NUM_WORDS)

# create word index
word_index = sp.get_word_index(texts)
print('created word index with total of {} unique tokens'.format(len(word_index)))
fname = PATH + 'word_index.json'
save_json(fname, word_index)
print('saved word index to file {}'.format(fname))

# get word labels
word_labels = sp.get_word_labels()
fname = PATH + 'word_labels.tsv'
save_txt(fname, word_labels)
print('saved word labels to file {}'.format(fname))

# create embedding matrix
emb_index = sp.load_pretrained_embedding(EMB_PATH)
print('loaded pretrained embedding containing {} unique tokens'.format(len(emb_index)))
emb_mat = sp.get_embedding_matrix()
print('created embedding matrix with {} rows and {} columns'.format(emb_mat.shape[0], emb_mat.shape[1]))
fname = PATH + 'emb_mat.hdf5'
save_array(fname, emb_mat, 'emb_mat')
print('saved embedding matrix to file {}'.format(fname))

# create sequences
# 1. language model
seqs = sp.get_language_model_seqs(texts, max_len=MAX_SEQ_LEN)
print('generated {} sequences with length {} for language model training'.format(seqs.shape[0], seqs.shape[1]))
fname = PATH + 'lang_model_seqs.hdf5'
save_array(fname, seqs, 'lang_model_seqs')
print('saved language model sequences to file {}'.format(fname))

# 2. retweet prediction
seqs = sp.get_seqs(tweets, max_len=MAX_SEQ_LEN)
print('generated {} sequences with length {} for retweet prediction'.format(seqs.shape[0], seqs.shape[1]))
fname = PATH + 'retpred_seqs.hdf5'
save_array(fname, seqs, 'retpred_seqs')
print('saved retweet prediction sequences to file {}'.format(fname))

# 3. irony detection
seqs = sp.get_seqs(irony, max_len=MAX_SEQ_LEN)
print('generated {} sequences with length {} for irony detection'.format(seqs.shape[0], seqs.shape[1]))
fname = PATH + 'irony_seqs.hdf5'
save_array(fname, seqs, 'irony_seqs')
print('saved irony detection sequences to file {}'.format(fname))

# 4. topic classification
seqs = sp.get_seqs(topic, max_len=MAX_SEQ_LEN)
print('generated {} sequences with length {} for topic classification'.format(seqs.shape[0], seqs.shape[1]))
fname = PATH + 'topic_seqs.hdf5'
save_array(fname, seqs, 'topic_seqs')
print('saved topic classification sequences to file {}'.format(fname))

# 5. offensive language detection
seqs = sp.get_seqs(offlang, max_len=MAX_SEQ_LEN)
print('generated {} sequences with length {} for offensive language detection'.format(seqs.shape[0], seqs.shape[1]))
fname = PATH + 'offlang_seqs.hdf5'
save_array(fname, seqs, 'offlang_seqs')
print('saved offensive language detection sequences to file {}'.format(fname))

# 6. valence classification
seqs = sp.get_seqs(valence, max_len=MAX_SEQ_LEN)
print('generated {} sequences with length {} for valence classification'.format(seqs.shape[0], seqs.shape[1]))
fname = PATH + 'valence_seqs.hdf5'
save_array(fname, seqs, 'valence_seqs')
print('saved valence classification sequences to file {}'.format(fname))

# 7. anger classification
seqs = sp.get_seqs(anger, max_len=MAX_SEQ_LEN)
print('generated {} sequences with length {} for anger classification'.format(seqs.shape[0], seqs.shape[1]))
fname = PATH + 'anger_seqs.hdf5'
save_array(fname, seqs, 'anger_seqs')
print('saved anger classification sequences to file {}'.format(fname))

# 8. joy classification
seqs = sp.get_seqs(joy, max_len=MAX_SEQ_LEN)
print('generated {} sequences with length {} for joy classification'.format(seqs.shape[0], seqs.shape[1]))
fname = PATH + 'joy_seqs.hdf5'
save_array(fname, seqs, 'joy_seqs')
print('saved joy classification sequences to file {}'.format(fname))

# 9. fear classification
seqs = sp.get_seqs(fear, max_len=MAX_SEQ_LEN)
print('generated {} sequences with length {} for fear classification'.format(seqs.shape[0], seqs.shape[1]))
fname = PATH + 'fear_seqs.hdf5'
save_array(fname, seqs, 'fear_seqs')
print('saved fear classification sequences to file {}'.format(fname))

# 10. sadness classification
seqs = sp.get_seqs(sadness, max_len=MAX_SEQ_LEN)
print('generated {} sequences with length {} for sadness classification'.format(seqs.shape[0], seqs.shape[1]))
fname = PATH + 'sadness_seqs.hdf5'
save_array(fname, seqs, 'sadness_seqs')
print('saved sadness classification sequences to file {}'.format(fname))
