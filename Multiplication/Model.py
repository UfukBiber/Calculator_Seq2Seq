import numpy as np
import tensorflow as tf
import Data_V2


Quantity_Of_Training_Data = 210000
Min_Value_Of_Training_Data = 0
Max_Value_Of_Training_Data = 500000

Characters = Data_V2.Characters

Enc_Inp, Dec_Inp, Output = Data_V2.PrepareData(Quantity_Of_Training_Data, Min_Value_Of_Training_Data, Max_Value_Of_Training_Data)



np.save("EncInp", Enc_Inp)
np.save("DecInp", Dec_Inp)
np.save("Output", Output)

class MyCallBack(tf.keras.callbacks.Callback):
    def __init__(self):
        self.acc = 0.75
    def on_epoch_end(self, epoch, logs={}):
        if logs.get("accuracy") > self.acc:
            print("\n\nReached %f percent accuracy"%self.acc)
            print("Model is saved")
            self.acc += 0.025
            self.model.save("my_model_2")
        if logs.get("accuracy")>0.995:
            self.model.stop_training = True
            self.model.save("my_model_2")

embedding = tf.keras.layers.Embedding(len(Characters)+1, 32)  

callback = MyCallBack()




EncInp = tf.keras.layers.Input(shape= (Enc_Inp.shape[-1]), name = "Encoder_Input")
encOut = embedding(EncInp)
encOut, Enc_State_h, Enc_state_c = tf.keras.layers.LSTM(512, return_state = True)(encOut)


DecInp = tf.keras.layers.Input(shape=(Dec_Inp.shape[-1]), name = "Decoder_Input")
decOut = embedding(DecInp)
decOut = tf.keras.layers.LSTM(512, return_sequences = True)(decOut, initial_state = [Enc_State_h, Enc_state_c])
decOut, _, _ = tf.keras.layers.LSTM(256, return_sequences = True, return_state = True)(decOut)
# decOut = tf.keras.layers.Dropout(0.4)(decOut)
decOut = tf.keras.layers.Dense(256, activation = "relu")(decOut)
decOut = tf.keras.layers.Dense(len(Characters)+1, activation = "softmax")(decOut)

model = tf.keras.models.Model([EncInp, DecInp], decOut)
model.compile(optimizer = "adam", loss = "sparse_categorical_crossentropy", metrics = ["accuracy"])
print(model.summary())


model.fit([Enc_Inp, Dec_Inp], Output, epochs = 1000, callbacks = [callback] )