import tensorflow as tf
import numpy as np
import Data

Quantity_Of_Training_Data = 50000
Min_Value_Of_Training_Data = 10000
Max_Value_Of_Training_Data = 500000



Enc_Inp, Dec_Inp, Output = Data.PrepareData(Quantity_Of_Training_Data, Min_Value_Of_Training_Data, Max_Value_Of_Training_Data)



np.save("EncInp", Enc_Inp)
np.save("DecInp", Dec_Inp)
np.save("Output", Output)



class MyCallBack(tf.keras.callbacks.Callback):
    def __init__(self):
        self.acc = 0.65
    def on_epoch_end(self, epoch, logs={}):
        if logs.get("accuracy") > self.acc:
            print("\n\nReached %f percent accuracy"%self.acc)
            print("Model is saved")
            self.acc += 0.025
            self.model.save("my_model")
        if logs.get("accuracy")>0.995:
            self.model.stop_training = True
            self.model.save("my_model")


model = tf.keras.Sequential([ 
    tf.keras.layers.Embedding(len(Data.Characters)+1, 32, mask_zero = True),
    tf.keras.layers.GRU(256),
    tf.keras.layers.RepeatVector(Output.shape[-1]),
    tf.keras.layers.GRU(256, return_sequences = True),
    tf.keras.layers.GRU(128, return_sequences = True),
    tf.keras.layers.Dense(128, activation = "relu"),
    tf.keras.layers.Dense(len(Data.Characters), activation = "softmax")
])

callback = MyCallBack()


model.compile(optimizer = "adam", loss = "sparse_categorical_crossentropy", metrics = ["accuracy"])
model.fit(Enc_Inp, Output, epochs = 1000, callbacks = [callback], validation_split = 0.1)

