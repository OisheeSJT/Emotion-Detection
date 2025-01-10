from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import numpy as np

#In here we Load the preprocessed data
X_train = np.load('X_train.npy')
y_train = np.load('y_train.npy')

#Define the model and its architecture
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#Training the model with X_train,y_train.
#set epochs as per your requirement. the more the better. Note it will take longer to train with more epochs
model.fit(X_train.reshape(-1, 48, 48, 1), y_train, epochs=30, batch_size=32)

#Save the trained model
model.save('Emotion_Detection.h5')

print("Model training complete!")
