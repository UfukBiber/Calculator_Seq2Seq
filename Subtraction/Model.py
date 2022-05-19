import numpy as np
import tensorflow as tf
import Data


Quantity_Of_Training_Data = 180000
Min_Value_Of_Training_Data = 10000
Max_Value_Of_Training_Data = 500000

Characters = Data.Characters

# Enc_Inp, Dec_Inp, Output = Data.PrepareData(Quantity_Of_Training_Data, Min_Value_Of_Training_Data, Max_Value_Of_Training_Data)



# np.save("EncInp", Enc_Inp)
# np.save("DecInp", Dec_Inp)
# np.save("Output", Output)

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



callback = MyCallBack()


Enc_Inp = np.load("EncInp.npy")
Dec_Inp = np.load("DecInp.npy")
Output = np.load("Output.npy")

# embedding = tf.keras.layers.Embedding(len(Characters)+1, 16, mask_zero = True)  




# EncInp = tf.keras.layers.Input(shape= (Enc_Inp.shape[-1]), name = "Encoder_Input")
# encOut = embedding(EncInp)
# encOut, Enc_State_for, Enc_state_back = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(128, return_state = True))(encOut)
# states = tf.keras.layers.Concatenate()([Enc_State_for, Enc_state_back])

# DecInp = tf.keras.layers.Input(shape=(Dec_Inp.shape[-1]), name = "Decoder_Input")
# decOut = embedding(DecInp)
# decOut, _ = tf.keras.layers.GRU(256, return_sequences = True, return_state = True)(decOut, initial_state = states)
# decOut = tf.keras.layers.Dropout(0.4)(decOut)
# decOut = tf.keras.layers.Dense(len(Characters)+1, activation = "softmax")(decOut)

# model = tf.keras.models.Model([EncInp, DecInp], decOut)
# model.compile(optimizer = "adam", loss = "sparse_categorical_crossentropy", metrics = ["accuracy"])


model = tf.keras.models.load_model("my_model")

model.fit([Enc_Inp, Dec_Inp], Output, epochs = 1000, callbacks = [callback], validation_split = 0.1)
