import tensorflow as tf
import os
import numpy as np
import Data_2


Quantity_Of_Training_Data = 70000
Min_Value_Of_Training_Data = 0
Max_Value_Of_Training_Data = 500000

Characters = Data_2.Characters


# Inp, decInp, Out = Data_2.PrepareData(Quantity_Of_Training_Data, Min_Value_Of_Training_Data, Max_Value_Of_Training_Data)

# np.save("Inp", Inp)
# np.save("decInp", decInp)
# np.save("output", Out)

class MyCallBack(tf.keras.callbacks.Callback):
    def __init__(self):
        self.acc = 0.6
    def on_epoch_end(self, epoch, logs={}):
        if logs.get("accuracy") > self.acc:
            print("\n\nReached %f percent accuracy"%self.acc)
            print("Model is saved")
            self.acc += 0.05
            self.model.save("my_model")
        if logs.get("accuracy")>0.995:
            self.model.stop_training = True
            self.model.save("my_model")

embedding = tf.keras.layers.Embedding(len(Characters)+1, 64)  

callback = MyCallBack()

Inp = np.load("Inp.npy")
decInp = np.load("decInp.npy")
Out = np.load("output.npy")


EncInp = tf.keras.layers.Input(shape= (Inp.shape[-1]))
encOut = embedding(EncInp)
encOut, Enc_State_h, Enc_state_c = tf.keras.layers.LSTM(256, return_state = True)(encOut)

DecInp = tf.keras.layers.Input(shape=(Out.shape[-1]))
decOut = embedding(DecInp)
decOut, _, _ = tf.keras.layers.LSTM(256, return_sequences = True, return_state = True)(decOut, initial_state = [Enc_State_h, Enc_state_c])
decOut = tf.keras.layers.Dense(len(Characters)+1, activation = "softmax")(decOut)

model = tf.keras.models.Model([EncInp, DecInp], decOut)
model.compile(optimizer = "adam", loss = "sparse_categorical_crossentropy", metrics = ["accuracy"])



model.fit([Inp, decInp], Out, epochs = 1000, callbacks = [callback], validation_split = 0.1 )