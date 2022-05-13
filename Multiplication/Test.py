import tensorflow as tf
import Data_V2



model = tf.keras.models.load_model("my_model_2")

EncInp, DecInp, Out = Data_V2.PrepareData(3000, 0, 500000)

model.evaluate([EncInp, DecInp], Out)