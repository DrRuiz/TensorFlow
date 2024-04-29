# Import TensorFlow into your program to get started:
import tensorflow as tf

# Load and prepare the MNIST dataset.
# The pixel values of the images range from 0 through 255. 
# Scale these values to a range of 0 to 1 by dividing the values by 255.0. 
# This also converts the sample data from integers to floating-point numbers:
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
                # to tell us if it is black or white since all the pixels will now be from 0 to 1
x_train, x_test = x_train / 255.0, x_test / 255.0 

# Build a tf.keras.Sequential model: (the only thing we need to change in the assignment)
# -- This is building the neuronet for us
# -- It asks us what type of shape we want our neuronet to be
model = tf.keras.models.Sequential([
  # image 28 (height) x 28 (height) x 1 (channel)
  # Convolution with 5 x 5 kernel + 2 padding: 28 x 28 x 6
  # Sigmoid
  tf.keras.layers.Conv2D(filters = 6, kernel_size = [5, 5], strides = 1, padding = "same", activation = 'sigmoid', input_shape = (28, 28, 1)), 
  # Pool with 2 x 2 average kernel + 2 stride: 14 x 14 x 6
  tf.keras.layers.AveragePooling2D(pool_size = [2, 2], strides = 2),
  # Convolution with 5 x 5 kernel (no pad): 10 x 10 x 16
  # Sigmoid
  tf.keras.layers.Conv2D(filters = 16, kernel_size = [5, 5], strides = 1, padding = "valid", activation = 'sigmoid'),
  # Pool with 2 x 2 average kernel + 2 stride: 5 x 5 16
  tf.keras.layers.AveragePooling2D(pool_size = [2, 2], strides = 2),
  # Flatten
  tf.keras.layers.Flatten(),
  # Dense: 120 fully connected neurons
  # Sigmoid
    tf.keras.layers.Dense(120, activation = 'sigmoid'),
  # Dense: 84 fully connected neurons
  # Sigmoid
    tf.keras.layers.Dense(84, activation = 'sigmoid'),
  # Dense 10 fully connected neurons
    tf.keras.layers.Dense(10, activation = 'sigmoid')
  # Output: 1 or 10 classes (which number we think this is)
])

# For each example, the model returns a vector of logits or log-odds scores, one for each class.
predictions = model(x_train[:1]).numpy()
predictions

# The tf.nn.softmax function converts these logits to probabilities for each class:
tf.nn.softmax(predictions).numpy()

# Define a loss function for training using losses.SparseCategoricalCrossentropy:
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# This untrained model gives probabilities close to random (1/10 for each class), 
loss_fn(y_train[:1], predictions).numpy()

# Before you start training, configure and compile the model using Keras Model.compile. 
model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

# ---- Train and evaluate your model

# Usd to adjust your model parameters and minimize the loss:
model.fit(x_train, y_train, epochs=5)

# To check the model's performance, usually on a validation set or test set.
test_loss, test_acc = model.evaluate(x_test,  y_test, verbose=2)
print(f"Test Accuracy: {test_acc*100:.2f}%")
print(f"Test loss: {test_loss*100:.2f}%")