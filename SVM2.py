import matplotlib.pyplot as plt
from sklearn import datasets, svm, metrics
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

digits = datasets.load_digits()

images_and_labels = list(zip(digits.images, digits.target))

# to apply a classifier on this data, we need to flatten the image: instead of an 8x8 matrix
# we have to use a one-dimensional array with 64 items
data = digits.images.reshape((len(digits.images), -1))

classifier = svm.SVC(gamma=0.001)

# 75% of the original dataset is for training
train_test_split = int(len(digits.images) * 0.75)

classifier.fit(data[:train_test_split], digits.target[:train_test_split])

# now predict the value of the digit on the remaining 25%
expected = digits.target[train_test_split:]
predicted = classifier.predict(data[train_test_split:])

confusionMatrix = confusion_matrix(expected, predicted, labels=classifier.classes_)
display = ConfusionMatrixDisplay(confusion_matrix=confusionMatrix, display_labels=classifier.classes_)
display.plot()
plt.show()

print('Accuracy:', accuracy_score(expected, predicted)*100)

# let's test on the last few images
plt.imshow(digits.images[-2], cmap='gray', interpolation='nearest')
print('Prediction for test image: ', classifier.predict(data[-2].reshape(1, -1)))

plt.show()
