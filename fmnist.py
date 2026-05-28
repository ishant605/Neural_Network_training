import tensorflow as tf
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
%matplotlib inline

# first import the dataset from keras
(X_train,y_train), (X_test,y_test) = keras.datasets.fashion_mnist.load_data()

# defining the class
classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# normalizing the dataset
X_train_scaled = X_train/255.0
X_test_scaled = X_test/255.0

# one hot encoding for classification problem
y_train_categorical = keras.utils.to_categorical(y_train, num_classes=10)
y_test_categorical = keras.utils.to_categorical(y_test,10)

# defining the model
def get_model():
  model = keras.Sequential([
      keras.Input(shape=(28,28)),
      keras.layers.Flatten(),
      keras.layers.Dense(100, activation='relu'),
      keras.layers.Dense(50, activation='relu'),
      keras.layers.Dense(10, activation='sigmoid')
  ])
  model.compile(
      optimizer='adam',
      loss='categorical_crossentropy',
      metrics=['accuracy']
  )
  return model

# training the model using GPU
with tf.device('/device:GPU:0'):
  start = time.time()
  model_gpu = get_model()
  model_gpu.fit(X_train_scaled,y_train_categorical,epochs=1)
  end = time.time()
  print(f'Time taken on GPU: {end-start}')

y_predict = model_gpu.predict(X_test_scaled)
y_predict_labels = [np.argmax(i) for i in y_predict]

# training the model using CPU
with tf.device('/device:CPU:0'):
  start = time.time()
  model_cpu = get_model()
  model_cpu.fit(X_train_scaled,y_train_categorical,epochs=1)
  end = time.time()
  print(f'Time taken on CPU: {end-start}')

y_predict = model_cpu.predict(X_test_scaled)
y_predict_labels = [np.argmax(i) for i in y_predict]









