import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from network import model, test_images, test_labels

probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])

predictions = probability_model.predict(test_images)

plt.figure(figsize=(10, 10))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(test_images[i])
    plt.colorbar()
    plt.xlabel('Predict: {}'.format(np.argmax(predictions[i])))
plt.show()