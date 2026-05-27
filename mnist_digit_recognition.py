import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

# first import the dataset using keras
(X_train,y_train),(X_test,y_test) = keras.datasets.mnist.load_data()

print(len(X_train))
print(len(X_test))

# matshow will show the first element
plt.matshow(X_train[0])

# let us check the shape of the data
print(X_train.shape)

# Centering the data is important for better accuracy
X_train = X_train/255.0
X_test = X_test/255.0

# now time to create a model
model = keras.Sequential([
  keras.layers.Flatten(input_shape=(28,28)),
  keras.layers.Dense(100, activation='relu'),
  keras.layers.Dense(10, activation='sigmoid')
])

# compile the model for output
model.compile(
  optimizer='adam',
  loss='sparse_categorical_crossentropy',
  metrics=[accuracy]
)

# fit the model after compilation
model.fit(
  X_train,
  y_train,
  epochs=10
)

# before prediction we will evaluate on the test dataset
model.evaluate(X_test,y_test)

# now prediction
y_predict = model.predict(X_test) # this will give the prediction value
y_predict_labels = [np.argmax(i) for i in y_predict] # this will give the output

# produce a confusion matrix
cm = tf.math.confusion_matrix(labels=y_test, predictions=y_predict_labels)

# heatmap for the confusion matrix
plt.figure(figsize = (10,7))
sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('True')
