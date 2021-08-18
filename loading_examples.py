import keras
import tensorflow as tf
import keras.backend as K

'''Specitify for training keras models'''
def specificityKeras(y_true, y_pred):
    true_negatives = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1 - y_true, 0, 1)))
    return true_negatives / (possible_negatives + K.epsilon())

leaky_relu = tf.nn.leaky_relu
specificity_keras = specificityKeras

model = keras.models.load_model('DeepICClassifier.h5',custom_objects = {'leaky_relu':leaky_relu,'specificityKeras':specificity_keras})
