import tensorflow as tf
import Data
import numpy as np



model = tf.keras.models.load_model("my_model")
print(model.summary())

EncInp, DecInp, Out = Data.PrepareData(2000, 0, 500000)

model.evaluate([EncInp, DecInp], Out)


