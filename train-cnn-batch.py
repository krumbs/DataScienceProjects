import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv
from scipy.ndimage.filters import uniform_filter1d
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Flatten
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D, AveragePooling1D, MaxPool1D
from keras import backend as K
from sklearn.metrics import confusion_matrix
from keras.optimizers import Adam


# Define Functions
def precision(y_true, y_pred):
    """ Precision metric:
    Only computes a batch-wise average of precision.
    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    prec = true_positives / (predicted_positives + K.epsilon())
    return prec


def recall(y_true, y_pred):
    """ Recall metric:
    Only computes a batch-wise average of recall.
    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def f1(y_true, y_pred):
    prec = precision(y_true, y_pred)
    recall_ = recall(y_true, y_pred)
    return 2*((prec*recall_)/(prec+recall_+K.epsilon()))


def batch_generator(x_train_, y_train_, batch_size=32):
    """
    Gives equal number of positive and negative samples, and rotates them randomly in time
    """
    half_batch = batch_size // 2
    x_batch = np.empty((batch_size, x_train_.shape[1], x_train_.shape[2]), dtype='float32')
    y_batch = np.empty((batch_size, y_train_.shape[1]), dtype='float32')

    yes_idx = np.where(y_train_[:, 0] == 1.)[0]
    non_idx = np.where(y_train_[:, 0] == 0.)[0]

    while True:
        np.random.shuffle(yes_idx)
        np.random.shuffle(non_idx)

        x_batch[:half_batch] = x_train_[yes_idx[:half_batch]]
        x_batch[half_batch:] = x_train_[non_idx[half_batch:batch_size]]
        y_batch[:half_batch] = y_train_[yes_idx[:half_batch]]
        y_batch[half_batch:] = y_train_[non_idx[half_batch:batch_size]]

        # Augment Data
        for i in range(half_batch):
            if i % 2 == 0:
                x_batch[i] = np.flip(x_batch[i], axis=0)            # flip data
            if i % 3 == 0:
                for kc, ki in enumerate(x_batch[i]):
                    x_batch[i][kc] += np.random.normal(0, 0.1)      # add noise
        yield x_batch, y_batch


# Fix random seed for reproducibility
np.random.seed(22)

prep = __import__('prepare-data-batchgen')
prep2 = __import__('prepare-data-2')


# Load the data
raw_data = read_csv('./kepler-labelled-time-series-data/exoTrain.csv')

labels, _, data = prep.prep_data(raw_data, norm_criteria=2)

seq_length = data.shape[1]

# Format the training and validation sets
x_train_full = data[:, 0:]
y_train_full = labels.values - 1.
s = int(np.ceil(x_train_full.shape[0]//1.5))
x_train = x_train_full[0:s, :]
y_train_T = y_train_full[0:s]
y_train = np.array([y_train_T]).T
del y_train_T
x_val = x_train_full[s:, :]
y_val_T = y_train_full[s:]
y_val = np.array([y_val_T]).T
del y_val_T
x_train = np.stack([x_train, uniform_filter1d(x_train, axis=1, size=200)], axis=2)
x_val = np.stack([x_val, uniform_filter1d(x_val, axis=1, size=200)], axis=2)
del data

# Select model 1 or 2
model_type = 2
if model_type == 1:
    # Model no.1
    model = Sequential()
    model.add(Conv1D(8, 32, activation='relu', input_shape=x_train.shape[1:]))
    model.add(BatchNormalization())
    model.add(Conv1D(16, 32, activation='relu'))
    model.add(AveragePooling1D(16, padding='same'))
    model.add(BatchNormalization())
    model.add(Conv1D(8, 64, activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv1D(16, 64, activation='relu'))
    model.add(GlobalAveragePooling1D())
    model.add(Dropout(0.25))
    model.add(Dense(1, activation='sigmoid'))
elif model_type == 2:
    # Model no.2
    model = Sequential()
    model.add(Conv1D(filters=8, kernel_size=11, activation='relu', input_shape=x_train.shape[1:]))
    model.add(MaxPool1D(strides=4))
    model.add(BatchNormalization())
    model.add(Conv1D(filters=16, kernel_size=11, activation='relu'))
    model.add(MaxPool1D(strides=4))
    model.add(BatchNormalization())
    model.add(Conv1D(filters=32, kernel_size=11, activation='relu'))
    model.add(MaxPool1D(strides=4))
    model.add(BatchNormalization())
    model.add(Conv1D(filters=64, kernel_size=11, activation='relu'))
    model.add(MaxPool1D(strides=4))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer=Adam(3e-3), loss='binary_crossentropy', metrics=["accuracy", precision, f1, recall])
print(model.summary())

# Save best weights to file
filepath = "bestweights.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

# Add weighting
category_weights = {0: 1., 1: 12}
# Fit the model
print('y_train shape: {}'.format(y_train.shape))

history = model.fit_generator(batch_generator(x_train, y_train, 32), validation_data=(x_val, y_val), callbacks=callbacks_list,
                                                class_weight=category_weights, verbose=0, epochs=10, steps_per_epoch=x_train.shape[1] // 32)

model.load_weights("bestweights.hdf5")

score = model.evaluate(x_val, y_val)
print(score)
print('RESULTS...')
predictions = model.predict(x_val)
predictions = np.round(predictions)
output_pair = [(x[0], y_val[xi]) for xi, x in enumerate(predictions)]
correct_sum = 0
total = len(output_pair)
for r in output_pair:
    if abs(r[1]-r[0]) < 0.5:
        correct_sum += 1
print("Confusion matrix")
cm = confusion_matrix(y_val, predictions)
print(cm)
tn, fp, fn, tp = cm.ravel()
print("tn, fp, fn, tp", tn, fp, fn, tp)
print("My validation accuracy", float(correct_sum)/total)
print("My validation precision", tp/(tp+fp))

# Make plots
plt.subplot(3, 1, 1)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')

plt.subplot(3, 1, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')

plt.subplot(3, 1, 3)
plt.plot(history.history['precision'])
plt.plot(history.history['val_precision'])
plt.title('model precision')
plt.ylabel('precision')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# Load the test data
raw_data_test = read_csv('./kepler-labelled-time-series-data/exoTest.csv')

labels_test, _, data_test = prep.prep_data(raw_data_test, norm_criteria=2)

x_test = data_test
x_test = np.stack([x_test, uniform_filter1d(x_test, axis=1, size=200)], axis=2)
y_test_T = labels_test - 1.
y_test = np.array([y_test_T]).T

print('RESULTS...')
predictions_2 = model.predict(x_test)
predictions_2 = np.round(predictions_2)
output_pair_2 = [(x[0], y_test[xi]) for xi, x in enumerate(predictions_2)]
correct_sum = 0
total = len(output_pair_2)
for r in output_pair_2:
    #print(r)
    if abs(r[1]-r[0]) < 0.5:
        correct_sum += 1
print("My accuracy", float(correct_sum)/total)

print("Confusion matrix")
cm = confusion_matrix(y_test, predictions_2)

print(cm)
tn, fp, fn, tp = cm.ravel()
print("tn, fp, fn, tp", tn, fp, fn, tp)

score = model.evaluate(x_test, y_test, verbose=1)
print('SCORE: \n', score)
print('test accuracy: ', score[1])
print('test precision: ', score[2])
print('test recall: ', tp/(tp+fn))
