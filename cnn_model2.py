import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Precision, Recall
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K

def weighted_binary_crossentropy(zero_weight=5.0, one_weight=1.0):
    def loss(y_true, y_pred):
        # Osiguranje da su y_pred između epsilon i 1 - epsilon
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # Izračunavanje ponderisane binarne cross-entropy
        bce = - (one_weight * y_true * K.log(y_pred) + zero_weight * (1 - y_true) * K.log(1 - y_pred))
        # Vraća se prosek gubitka
        return K.mean(bce)
    
    return loss


# Set some global parameters
img_size = (210, 210)  # Size of the input images
matrix_size = (21, 21)  # Size of the output matrix
# Function to load maze images and corresponding matrices
flag = 1
def load_data(input_dir, output_dir):
    images = []
    matrices = []
    
    # Loop through the input folder (images)
    for img_file in sorted(os.listdir(input_dir)):
        # Load the image and convert to array
        img_path = os.path.join(input_dir, img_file)
        img = load_img(img_path, target_size=img_size, color_mode='grayscale')
        img_array = img_to_array(img) / 255.0  # Normalize the image
        images.append(img_array)
    
    # Loop through the output folder (matrices)
    for matrix_file in sorted(os.listdir(output_dir)):
        # Load the matrix from text file
        matrix_path = os.path.join(output_dir, matrix_file)
        matrix = np.loadtxt(matrix_path, delimiter=',')
           
        matrices.append(matrix)
    
    return np.array(images), np.array(matrices)
# Load the training data
train_input_dir = 'dataset\\train\\data_x'
train_output_dir = 'dataset\\train\\data_y'
X_train, y_train = load_data(train_input_dir, train_output_dir)
# Load the test data
test_input_dir = 'dataset\\test\\test_x'
test_output_dir = 'dataset\\test\\test_y'
X_test, y_test = load_data(test_input_dir, test_output_dir)
# Build the CNN model
model = Sequential()
#//////////////////////////////////////////////////////////////////model
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(210, 210, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(512, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

# Flatten the results to feed into the fully connected layers
model.add(Flatten())
# Fully connected layer
model.add(Dense(512, activation='relu'))
#model.add(Dropout(0.5))
model.add(Dense(21 * 21, activation='sigmoid'))
model.add(tf.keras.layers.Reshape((21, 21)))
optimizer = Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
#model.compile(optimizer=optimizer, loss=weighted_binary_crossentropy(), metrics=['accuracy'])
# Train the model

model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_acc}')

# Save the model
model.save('modeli\\maze_cnn_model.h5')

# Load the model
#model = tf.keras.models.load_model('modeli\\maze_cnn_model.h5')
# Predict on a new image
new_image_path = 'dataset\\provera\\maze tt.png'
new_image = load_img(new_image_path, target_size=img_size, color_mode='grayscale')
new_image = img_to_array(new_image) / 255.0
new_image = np.expand_dims(new_image, axis=0)
# Get the predicted matrix
predicted_matrix = model.predict(new_image)
predicted_matrix = predicted_matrix.reshape((21, 21))
predicted_matrix = np.round(predicted_matrix).astype(int)
print(predicted_matrix)

plt.figure(figsize=(5,5))
plt.imshow(predicted_matrix, cmap='gray')
plt.xticks([]), plt.yticks([])
plt.grid(color='black', linewidth=2)
plt.show()