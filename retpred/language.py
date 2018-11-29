from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding, Dropout
from tensorflow.keras.regularizers import l2
from .utils.train import callbacks as std_callbacks
from .utils.train import ClassificationMetrics

def lstm_language_model(emb_mat, vocab_size=20001, emb_dim=200, seq_len=31, emb_dropout=0.25, lstm_units=128, l2_reg=1e-4, lstm_dropout=0.25, output_dropout=0.25):
    model = Sequential()
    model.add(Embedding(vocab_size, emb_dim, input_length=seq_len, weights=[emb_mat], name='word_embedding'))
    model.add(Dropout(emb_dropout, name='embedding_dropout'))
    model.add(LSTM(lstm_units,
        kernel_regularizer=l2(l2_reg),
        recurrent_regularizer=l2(l2_reg),
        bias_regularizer=l2(l2_reg),
        dropout=lstm_dropout,
        recurrent_dropout=lstm_dropout,
        name='lstm'))
    model.add(Dropout(output_dropout, name='output_dropout'))
    model.add(Dense(vocab_size, 
        activation='softmax', 
        kernel_regularizer=l2(l2_reg),
        bias_regularizer=l2(l2_reg),
        name='prediction'))
    model.compile(loss='sparse_categorical_crossentropy', 
            optimizer='adam',
            metrics=['accuracy'])
    return model

def callbacks(model_path, model_name, early_stopping=True):
    cbs = std_callbacks(model_path, model_name, early_stopping=early_stopping)
    metrics_logger = ClassificationMetrics()
    cbs.append(metrics_logger)
    return cbs
