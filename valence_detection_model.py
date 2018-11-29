from subprocess import call
from tensorflow.keras.utils import to_categorical
from retpred.transfer import transfer_model, callbacks
from retpred.utils.io import load_array
from retpred.utils.model import load_architecture, save_architecture
from retpred.utils.train import min_val_loss

# model constants
MODEL_PATH = '/storage/transfer_valence/'
MODEL_NAME = 'transfer_valence'
BASE_MODEL_PATH = '/storage/language_models/regular_lstm'
DATA_PATH = '/storage/'
DROPOUT = 0.0
EPOCHS = 50
BATCH_SIZE = 128
NUM_CLASSES = 7

# create storage folder
call(['mkdir', '-p', MODEL_PATH])

# load and prepare data
seqs = load_array(DATA_PATH + 'valence_seqs.hdf5', 'valence_seqs')
# cut first leading 0 in order to conform with language model sequence length
# TODO: retrain language model with 32-dimensional sequences
seqs = seqs[:,1:]
print('loaded {} input sequences with length {}'.format(seqs.shape[0], seqs.shape[1]))
labels = load_array(DATA_PATH + 'sentiment/valence_labels.hdf5', 'valence_labels')
labels = to_categorical(labels, num_classes=NUM_CLASSES)
print('loaded {} outputs with {} classes'.format(labels.shape[0], labels.shape[1]))

# load base model
model = load_architecture(BASE_MODEL_PATH + '.json')
model.load_weights(BASE_MODEL_PATH + '.hdf5')
print('loaded base model architecture and weights from {}'.format(BASE_MODEL_PATH))

# define callbacks
cbs = callbacks(MODEL_PATH, MODEL_NAME, early_stopping=False)

# load model
model = transfer_model(model, NUM_CLASSES, dropout=DROPOUT)
fname = MODEL_PATH + MODEL_NAME + '.json'
save_architecture(fname, model)
print('model architecture saved to file {}'.format(fname))

# create training and validation sets
x_trn = seqs[:1000]
y_trn = labels[:1000]
x_val = seqs[-1000:]
y_val = labels[-1000:]
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
# print results
logger = cbs[-1]
metrics = min_val_loss(logger)
print('ended model training with following results:')
print('loss: {}'.format(metrics['loss']))
print('validation loss: {}'.format(metrics['val_loss']))
print('accuracy: {}'.format(metrics['acc']))
print('validation accuracy: {}'.format(metrics['val_acc']))
