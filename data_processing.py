import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


data = pd.read_csv('fer2013.csv')

X = []
y = []


for index, row in data.iterrows():
    # Get pixel values and convert them into an array
    pixels = np.array(row['pixels'].split(), dtype='float32')
    image = pixels.reshape(48, 48)  # Reshape to 48x48 pixels
    X.append(image)
    y.append(row['emotion'])  # Append emotion label


X = np.array(X)
y = np.array(y)


X = X / 255.0

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


from keras.utils import to_categorical
y_train = to_categorical(y_train, num_classes=7)  # Assuming there are 7 emotions now
y_test = to_categorical(y_test, num_classes=7)

np.save('X_train.npy', X_train)
np.save('X_test.npy', X_test)
np.save('y_train.npy', y_train)
np.save('y_test.npy', y_test)

print("Data preprocessing complete!")
