# MNIST Dataset
[MNIST Dataset](http://yann.lecun.com/exdb/mnist/) with 60.000 handwritten digits.

# Specifications
Handwritten digits from 0 to 9, grayscale (0 - 255) and with a resolution of 28 x 28 pixels. 

# Alternatives
Install [tensorflow](https://www.tensorflow.org/install) with [pip](https://pypi.org/project/pip/):
```
pip install tensorflow
```
Now import the MNIST dataset with:
```python
import tensorflow as tf

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data() 
```
