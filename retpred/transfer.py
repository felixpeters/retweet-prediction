from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense
from tensorflow.keras.regularizers import l2
from .utils.train import callbacks as std_callbacks
from .utils.train import ClassificationMetrics

def transfer_model(model, num_classes, opt=Adam(), l2_reg=1e-4):
    model.pop()
    model.add(Dense(num_classes,
        activation='softmax',
        kernel_regularizer=l2(l2_reg),
        bias_regularizer=l2(l2_reg),
        name='transfer_prediction'))
    for layer in model[:-1]:
        layer.trainable = False
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy']) 
    return model

def callbacks(model_path, model_name, early_stopping=True):
    cbs = std_callbacks(model_path, model_name, early_stopping=early_stopping)
    metrics_logger = ClassificationMetrics()
    cbs.append(metrics_logger)
    return cbs
