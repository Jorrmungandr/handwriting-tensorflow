import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from inputProvider import inputs
from network import model, test_images

probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])

input_images = np.array(inputs)

predictions = probability_model.predict(input_images)

plt.figure(figsize=(10, 10))
for i in range(len(input_images)):
    plt.subplot(2, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(input_images[i])
    plt.xlabel('Predict: {}'.format(np.argmax(predictions[i])))
plt.show()

# Run server
# tensorflow_model_server   --rest_api_port=8501   --model_name=handrecog   --model_base_path="/tmp/handrecog" >server.log 2>&1