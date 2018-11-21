import operator
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

class SequencePreprocessor():
    """
    Class for preparing preprocessed texts into sequences and creating an
    embedding matrix.
    """

    def __init__(self, num_words=20000):
        self.word_index = {}
        self.embeddings_index = {}
        self.embedding_dimension = 0
        self.tokenizer = Tokenizer(num_words=num_words, filters='"”“#$%&()*+,/=@[]^_´`‘{|}~\t\n\\`"')

    def get_word_index(self, texts):
        """
        Creates the word index from the given texts.
        """
        self.tokenizer.fit_on_texts(texts)
        self.word_index = self.tokenizer.word_index
        return self.word_index
    
    def get_seqs(self, texts, max_len=32):
        """
        Transforms texts into sequences of word indices.
        Pads sequences so that they have equal length.
        Attention: Only callable after word index was generated.
        """
        sequences =  self.tokenizer.texts_to_sequences(texts)
        return pad_sequences(sequences, maxlen=max_len)

    def get_language_model_seqs(self, texts, max_len=32):
        """
        Creates (n-1) sequences for a text with n tokens, which can be fed into
        a language model.
        """
        seqs = self.tokenizer.texts_to_sequences(texts)
        input_seqs = []
        for seq in seqs:
            for i in range(1, len(seq)):
                n_gram_seq = seq[:i+1]
                input_seqs.append(n_gram_seq)
        input_seqs = np.array(pad_sequences(input_seqs, maxlen=max_len))
        return input_seqs

    def get_word_labels(self):
        """
        Return words ordered by number of occurrences in texts which were used
        for creating word index.
        """
        sorted_words = sorted(self.tokenizer.word_index.items(), key=operator.itemgetter(1))
        sorted_words = ['<unknown>'] + [w[0] for w in sorted_words]
        return sorted_words
    
    def load_pretrained_embedding(self, fname):
        """
        Loads a pretrained embeddings index from the given file.
        Assumes a file with one line per embedding, starting with word and
        followed by coefficients (separated by spaces).
        """
        f = open(fname, encoding='utf-8')
        coefs = [] 
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            self.embeddings_index[word] = coefs
        f.close()
        self.embedding_dimension= len(coefs)
        return self.embeddings_index
    
    def get_embedding_matrix(self):
        """
        Creates embedding matrix from word and embeddings indices.
        """
        embedding_matrix = np.zeros((len(self.word_index) + 1, self.embedding_dimension)) 
        for word, i in self.word_index.items():
            embedding_vector = self.embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found will stay all-zeros
                embedding_matrix[i] = embedding_vector
        return embedding_matrix
