import pandas as pd
import confusion_mat as cfm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np
import os
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD
from lenet import LeNet
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
data = pd.read_pickle('Data_32_ostu.pkl')
labels = pd.read_pickle('Labels_32_ostu.pkl')

trainX, testX, trainY, testY = train_test_split(data, labels, test_size=.25, random_state=1)


def step_decay(epoch):
    init_lr = 0.01
    factor = 0.5
    dropEvery = 5
    alpha = init_lr * (factor ** np.floor((1 + epoch) / dropEvery))
    return float(alpha)


fname = os.path.sep.join(['E:/Code/htmljscss/' + "val_loss_32_ostu.hdf5"])
checkpoint = ModelCheckpoint(filepath=fname, monitor='val_loss', save_best_only=True, verbose=1)
callbacks = [checkpoint]
print('[INFO] compiling model...')
opt = SGD(learning_rate=0.5, decay=0.5 / 10, momentum=0.9, nesterov=True)
model = LeNet.build(height=32, width=32, depth=1, classes=10)
model.compile(loss='categorical_crossentropy', optimizer=opt,
              metrics=['accuracy'])

print('[INFO] training network...')
H = model.fit(trainX, trainY, validation_data=(testX, testY),
              batch_size=128, epochs=10, callbacks=callbacks, verbose=1)
model.save('E:/Code/htmljscss/model_32_ostu.hdf5')

print('[INFO] evaluating network...')
predictions = model.predict(testX, batch_size=128)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1)))

np.set_printoptions(precision=1)
classLabels = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']

# Plot non-normalized confusion matrix
cfm.plot_confusion_matrix(testY.argmax(axis=1), predictions.argmax(axis=1), classes=classLabels)
plt.show()

plt.style.use('ggplot')
plt.figure()
plt.plot(np.arange(0, 10), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, 10), H.history['val_accuracy'], label='val_acc')
plt.title('Training Accuracy')
plt.xlabel('Epoch #')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.style.use('ggplot')
plt.figure()
plt.plot(np.arange(0, 10), H.history['loss'], label='train_loss')
plt.plot(np.arange(0, 10), H.history['val_loss'], label='val_loss')

plt.title('Training Loss ')
plt.xlabel('Epoch #')
plt.ylabel('Loss')
plt.legend()
plt.show()

