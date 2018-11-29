from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2
from .utils.train import callbacks as std_callbacks
from .utils.train import ClassificationMetrics

def transfer_model(model, num_classes, opt=Adam(), l2_reg=1e-4, dropout=0.0):
    model.pop()
    model.pop()
    model.add(Dropout(dropout, name='output_dropout'))
    model.add(Dense(num_classes,
        activation='softmax',
        kernel_regularizer=l2(l2_reg),
        bias_regularizer=l2(l2_reg),
        name='transfer_output'))
    for layer in model.layers[:-2]:
        layer.trainable = False
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy']) 
    return model

def callbacks(model_path, model_name, early_stopping=True):
    cbs = std_callbacks(model_path, model_name, early_stopping=early_stopping)
    metrics_logger = ClassificationMetrics()
    cbs.append(metrics_logger)
    return cbs
