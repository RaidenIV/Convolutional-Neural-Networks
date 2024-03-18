import tensorflow as tf
from keras import datasets, layers, models
import matplotlib.pyplot as plt

# Load the CIFAR-10 dataset and split it into training and testing sets
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize the images by making their pixel values between 0 and 1 for better training performance
train_images, test_images = train_images / 255.0, test_images / 255.0

# Define the class names for the CIFAR-10 dataset
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# Index of the image to be shown, change this value to see different images
IMG_INDEX = 4

# Display an image from the training set
plt.imshow(train_images[IMG_INDEX], cmap=plt.cm.binary)
plt.xlabel(class_names[train_labels[IMG_INDEX][0]])
plt.show()

# Start building the convolutional neural network model
model = models.Sequential()
# Add a 2D convolution layer with 32 filters, a 3x3 kernel, and ReLU activation function. 
# This is the first layer so input_shape is specified.
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
# Add a max pooling layer to reduce spatial dimensions of the output volume
model.add(layers.MaxPooling2D((2, 2)))
# Add another 2D convolution layer with 64 filters
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# Add another max pooling layer
model.add(layers.MaxPooling2D((2, 2)))
# Add a third 2D convolution layer with 64 filters
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

# Print the model summary to see the architecture so far
model.summary()

# Flatten the output of the last convolution layer to a 1D array to feed into the dense layer
model.add(layers.Flatten())
# Add a dense layer with 64 units and ReLU activation function
model.add(layers.Dense(64, activation='relu'))
# Add a final dense layer with 10 units for each class
model.add(layers.Dense(10))

# Print the complete model summary to see the full architecture
model.summary()

# Compile the model specifying the optimizer, loss function, and metrics to monitor
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train the model with the training data, also specifying the number of epochs and validation data
history = model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))

# Evaluate the model's performance on the test set
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

# Print the test accuracy
print(test_acc)