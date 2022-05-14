import tensorflow as tf
import Data

model = tf.keras.models.load_model("my_model_2")

EncInp, DecInp, Out = Data.PrepareData(2000, 0, 500000)

model.evaluate([EncInp, DecInp], Out)

