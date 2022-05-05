import numpy as np
import tensorflow as tf
import random
import os


MaxNumber = 500000
Quantity = 150000
InpLength = len(str(MaxNumber))
OutLength = len(str(2*MaxNumber-1)) + 1
Characters = "0123456789+-_"
Characters2Numbers = {c:i for i,c in enumerate(list(Characters))}
Numbers2Characters = {i:c for i,c in enumerate(list(Characters))}



class MyCallBack(tf.keras.callbacks.Callback):
    def __init__(self):
        self.acc = 0.6
        self.time = 3
    def on_epoch_end(self, epoch, logs={}):
        if logs.get("accuracy") > self.acc:
            print("\n\nReached %f percent accuracy\n\n"%self.acc)
            print("\nModel %i is saved"%self.time)
            self.acc += 0.05
            self.model.save("my_model_%i"%self.time)
            self.time += 1
        if logs.get("accuracy")>0.98:
            self.model.save("finalModel")
            self.model.stop_training = True
            

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
            outputAddition = str(x+y)
            outputSubtraction = str(x-y)
            if len(outputAddition) < OutLength:
                outputAddition = "_"*(OutLength-len(outputAddition)) + outputAddition
            if len(outputSubtraction) < OutLength:
                outputSubtraction = "_"*(OutLength-len(outputSubtraction)) + outputSubtraction
            x = str(x)
            y = str(y)
            if len(x) < InpLength:
                x = "_"*(InpLength-len(x)) + x
            if len(y) < InpLength:
                y = "_"*(InpLength-len(y)) + y
            Inp.append(VectorizeString("%s+%s"%(x,y)))
            Out.append(VectorizeString(outputAddition))
            Inp.append(VectorizeString("%s-%s"%(x, y)))
            Out.append(VectorizeString(outputSubtraction))
        print("\t Preparing the data : %i/%i"%(len(Inp), quantity), end="\r")
    return np.reshape(np.asarray(Inp, dtype = np.float32), (quantity, 2*InpLength+1, 1)), np.reshape(np.asarray(Out, dtype = np.float32), (quantity, OutLength, 1))
def notInclude(data, array):
    if data not in array:
        return True
    return False


if tf.test.is_gpu_available():
    print("\nGpu is being used.")
else:
    print("\nCpu is being used.")
print("\n")
if os.path.exists("Inp.npy"):
    Inp = np.load("Inp.npy")
    Out = np.load("Out.npy")
else:
    Inp, Out = PrepareData(Quantity)
    np.save("Inp.npy", Inp)
    np.save("Out.npy", Out)

if os.path.exists("my_model_2"):
    model = tf.keras.models.load_model("my_model_2")
    model.fit(Inp, Out, epochs = 100, validation_split=0.1, callbacks = [callback])
    # y = model.predict(np.reshape(np.array(VectorizeString("__1100-__1834")), (1, 13, 1)))
    # y = np.squeeze(y)
    # y = np.argmax(y, axis=1)
    # print(1100-1834)
    # for i in y:
    #     print(Numbers2Characters[i])
else:
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(256,  dtype = tf.float32),
        tf.keras.layers.RepeatVector(OutLength),
        tf.keras.layers.LSTM(128, return_sequences = True),
        tf.keras.layers.Dense(64, activation = "relu"),
        tf.keras.layers.Dense(len(Characters), activation = "softmax")
    ])
    model.compile(loss = "sparse_categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"])

    model.fit(Inp, Out, epochs = 20, validation_split = 0.1)
    model.save("my_model")


