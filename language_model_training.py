import random
from subprocess import call
from retpred.utils.io import load_array
from retpred.utils.model import save_architecture
from retpred.utils.train import min_val_loss
from retpred.language import lstm_language_model, callbacks

# set model parameters
MODEL_PATH = '/storage/language_models/'
MODEL_NAME = 'regular_lstm'
LSTM_UNITS = 128
EMB_DROPOUT = 0.0
LSTM_DROPOUT = 0.0
OUTPUT_DROPOUT = 0.0
EPOCHS = 100
BATCH_SIZE = 1024

# create storage folder
call(['mkdir', '-p', MODEL_PATH])

# load and prepare data
seqs = load_array('/storage/lang_model_seqs.hdf5', 'lang_model_seqs')
random.seed(2018)
random.shuffle(seqs)
inputs, labels = seqs[:,:-1], seqs[:,-1]
print('loaded {} input sequences with length {}'.format(inputs.shape[0], inputs.shape[1]))
print('loaded {} outputs'.format(labels.shape[0]))
emb_mat = load_array('/storage/emb_mat.hdf5', 'emb_mat')
print('loaded embedding matrix with {} words and {} dimensions'.format(emb_mat.shape[0], emb_mat.shape[1]))

# define callbacks
cbs = callbacks(MODEL_PATH, MODEL_NAME)

# load model
model = lstm_language_model(emb_mat, 
        emb_dropout=EMB_DROPOUT,
        lstm_units=LSTM_UNITS,
        lstm_dropout=LSTM_DROPOUT,
        output_dropout=OUTPUT_DROPOUT)
fname = MODEL_PATH + MODEL_NAME + '.json'
save_architecture(fname, model)
print('model archicture saved to file {}'.format(fname))

# define training and validation data
x_trn = inputs[:-10000]
y_trn = labels[:-10000]
x_val = inputs[-10000:]
y_val = labels[-10000:]
print('created training set with {} examples'.format(x_trn.shape[0]))
print('created validation set with {} examples'.format(x_val.shape[0]))

# train model
model.fit(x_trn,
        y_trn,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        verbose=2,
        callbacks=cbs,
        validation_data=(x_val, y_val))

# print results
logger = cbs[-1]
metrics = min_val_loss(logger)
print('ended model training with following results:')
print('loss: {}'.format(metrics['loss']))
print('validation loss: {}'.format(metrics['val_loss']))
print('accuracy: {}'.format(metrics['acc']))
print('validation accuracy: {}'.format(metrics['val_acc']))
