import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
x_train = x_train.reshape(-1, 784)
x_test = x_test.reshape(-1, 784)
encoding_dim = 64
input_img = Input(shape=(784,))
encoded = Dense(256, activation='relu')(input_img)
encoded = Dense(128, activation='relu')(encoded)
encoded = Dense(encoding_dim, activation='relu')(encoded)
decoded = Dense(128, activation='relu')(encoded)
decoded = Dense(256, activation='relu')(decoded)
decoded = Dense(784, activation='sigmoid')(decoded)
autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.fit(
    x_train, x_train,
    epochs=50,
    batch_size=256,
    shuffle=True,
    validation_data=(x_test, x_test)
)
encoder = Model(input_img, encoded)
encoded_imgs = encoder.predict(x_test)
index = 0
encoded_image = encoded_imgs[index]
encoded_image_reshaped = encoded_image.reshape(8, 8)
n = 10 
print("Shape of original images:", x_test.shape)
print("Shape of encoded representations:", encoded_imgs.shape)
num_reduced_features = encoded_imgs.shape[1]
print("Number of reduced features:", num_reduced_features)
plt.figure(figsize=(20, 6))
for i in range(n):
    ax = plt.subplot(3, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.title("Original")
    plt.gray()
    ax.axis('off')
    ax = plt.subplot(3, n, i + 1 + 2 * n)
    encoded_image = encoded_imgs[i].reshape(8, 8) 
    plt.imshow(encoded_image, cmap='viridis')
    plt.title(f"Encoded {i}")
    ax.axis('off')
    ax = plt.subplot(3, n, i + 1 + n)
    plt.imshow(autoencoder.predict(x_test[i].reshape(1, 784)).reshape(28, 28))
    plt.title("Predicted")
    ax.axis('off')
plt.tight_layout()
plt.show()
