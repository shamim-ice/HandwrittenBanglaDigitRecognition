import pandas as pd
import confusion_mat as cfm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

data = pd.read_pickle('Data_KNN.pkl')
labels = pd.read_pickle('Labels_KNN.pkl')

trainX, testX, trainY, testY = train_test_split(data, labels, test_size=.25, random_state=1)
model = KNeighborsClassifier(n_neighbors=200)
H = model.fit(trainX, trainY)
# acc=model.score(testX,testY)
# print("[INFO] raw pixel accuracy: {:.2f}%".format(acc * 100))
predictions = model.predict(testX)
print(classification_report(testY, predictions))

np.set_printoptions(precision=1)
classLabels = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']

# Plot non-normalized confusion matrix
cfm.plot_confusion_matrix(testY, predictions, classes=classLabels)
plt.show()
