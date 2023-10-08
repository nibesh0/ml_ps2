import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from face import Face  

X = np.load('X.npy')
Y = np.load('Y.npy')

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

X_train = X_train.reshape(X_train.shape[0], 200, 200, 1)
X_test = X_test.reshape(X_test.shape[0], 200, 200, 1)

model = tf.keras.models.load_model('model.h5')

predictions = model.predict(X_test)

threshold = 0.5
binary_predictions = (predictions > threshold).astype(int)

accuracy = accuracy_score(y_test, binary_predictions)
print(f'Accuracy: {accuracy * 100:.2f}%')
