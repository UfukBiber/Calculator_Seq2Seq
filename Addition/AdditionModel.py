import numpy as np
import tensorflow as tf
import random
import os
### Total 110 epochs

MaxNumber = 500000
Quantity = 50000
InpLength = len(str(MaxNumber))
OutLength = len(str(2*MaxNumber-1))
Characters = "0123456789+_"
Characters2Numbers = {c:i for i,c in enumerate(list(Characters))}
Numbers2Characters = {i:c for i,c in enumerate(list(Characters))}

class MyCallBack(tf.keras.callbacks.Callback):
    def __init__(self):
        self.acc = 0.50
        self.repetition = 1
    def on_epoch_end(self, epoch, logs={}):
        if self.repetition % 5 == 0:
            self.model.save("my_model_2")
        self.repetition+=1
            
callback = MyCallBack()

def VectorizeString(string):
    vectorizedString = []
    string = list(string)
    for chr in string:
        vectorizedString.append(Characters2Numbers[chr])
    return vectorizedString

def PrepareData(quantity):
    Inp = []
    Out = []
    allData = []
    while len(Inp) < quantity:
        x = random.randint(0, MaxNumber)
        y = random.randint(0, MaxNumber)
        if notInclude([x, y], allData):
            allData.append([x, y])
            output = str(x+y)
            if len(output) < OutLength:
                output = "_"*(OutLength-len(output)) + output
            x = str(x)
            y = str(y)
            if len(x) < InpLength:
                x = "_"*(InpLength-len(x)) + x
            if len(y) < InpLength:
                y = "_"*(InpLength-len(y)) + y
            Inp.append(VectorizeString("%s+%s"%(x,y)))
            Out.append(VectorizeString(output))
        print("\t Preparing the data : %i/%i"%(len(Inp), quantity), end="\r")
    return tf.reshape(tf.constant(Inp, dtype = tf.float32), (quantity, 2*InpLength+1, 1)), tf.reshape(tf.constant(Out, dtype = tf.float32), (quantity, OutLength, 1))
def notInclude(data, array):
    data_2 = [data[1], data[0]]
    if data_2 not in array and data not in array:
        return True
    return False

if tf.test.is_gpu_available():
    print("\nGpu is being used.")
else:
    print("\nCpu is being used.")
if os.path.exists("Inp.npy"):
    Inp = np.load("Inp.npy")
    Out = np.load("Out.npy")
else:
    Inp, Out = PrepareData(Quantity)
    np.save("Inp.npy", Inp.numpy())
    np.save("Out.npy", Out.numpy())

if os.path.exists("my_model"):
    model = tf.keras.models.load_model("my_model")
    model.fit(Inp, Out, epochs = 75, validation_split=0.1, callbacks = [callback])
else:
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(256,  dtype = tf.float32),
        tf.keras.layers.RepeatVector(OutLength),
        tf.keras.layers.LSTM(128, return_sequences = True),
        tf.keras.layers.Dense(64, activation = "relu"),
        tf.keras.layers.Dense(len(Characters), activation = "softmax")
    ])
    model.compile(loss = "sparse_categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"])

    model.fit(Inp, Out, epochs = 75, validation_split = 0.1, callbacks = [callback])
    model.save("my_model")


