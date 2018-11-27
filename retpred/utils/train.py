from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
import numpy as np

def callbacks(model_path, model_name, early_stopping=False):
    cbs = []
    fname = model_path + model_name + '.hdf5'
    mc = ModelCheckpoint(fname, verbose=1, save_best_only=True, save_weights_only=True)
    cbs.append(mc)
    if early_stopping:
        es = EarlyStopping(patience=3, verbose=1)
        cbs.append(es)
    return cbs

def min_val_loss(metrics):
    i = np.argmin(metrics.val_losses)
    res = {
        "loss": metrics.losses[i],
        "val_loss": metrics.val_losses[i],
        "acc": metrics.accs[i],
        "val_acc": metrics.val_accs[i],
    }
    return res

class ClassificationMetrics(Callback):
    def on_train_begin(self, logs={}):
        print('{"chart": "loss", "axis": "epochs"}')
        print('{"chart": "val_loss", "axis": "epochs"}')
        print('{"chart": "acc", "axis": "epochs"}')
        print('{"chart": "val_acc", "axis": "epochs"}')
        self.losses = []
        self.val_losses = []
        self.accs = []
        self.val_accs = []

    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.accs.append(logs.get('acc'))
        self.val_accs.append(logs.get('val_acc'))
        losses = self.losses
        val_losses = self.val_losses
        accs = self.accs
        val_accs = self.val_accs
        print('{"chart": "loss", "y": ' + str(losses[-1]) + ', "x": ' + str(len(losses)) + '}')
        print('{"chart": "val_loss", "y": ' + str(val_losses[-1]) + ', "x": ' + str(len(val_losses)) + '}')
        print('{"chart": "acc", "y": ' + str(accs[-1]) + ', "x": ' + str(len(accs)) + '}')
        print('{"chart": "val_acc", "y": ' + str(val_accs[-1]) + ', "x": ' + str(len(val_accs)) + '}')
