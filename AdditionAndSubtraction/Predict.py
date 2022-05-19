import tensorflow as tf
import Data
import numpy as np

Char2Num = Data.Characters2Numbers
Num2Char = Data.Numbers2Characters




model = tf.keras.models.load_model("my_model")
print(model.layers)




######## ENCODER #####
EncoderInput = model.inputs[0]
EncoderStates = model.layers[4].output

EncoderModel = tf.keras.models.Model(EncoderInput, EncoderStates)



####################
#### DECODER #######


DecInput = model.inputs[1]

DecOut = model.layers[2](DecInput)

DecoderInpState = tf.keras.layers.Input(shape = (512, ))
DecoderGru, DecoderOutState = model.layers[5](DecOut, initial_state = DecoderInpState)
DecoderGru, _ = model.layers[6](DecoderGru)
Dropout = model.layers[7](DecoderGru)
Output = model.layers[8](Dropout)
Output = model.layers[9](Output)



DecoderModel = tf.keras.models.Model([DecInput, DecoderInpState], [Output, DecoderOutState])

####################

def Predict(Inp):
    if len(Inp) < 13:
        Inp += "_"*(13-len(Inp))
    Inp = list(Inp)
    VectorInp = tf.expand_dims(tf.constant([Char2Num[i] for i in Inp]), axis = 0)
    states = EncoderModel(VectorInp)
    decInp = np.zeros((1, 1))
    decInp[:, 0] = Char2Num["s"]
    result = []
    step = 0
    while step < 9:
        output, states = DecoderModel([decInp, states])
        output = np.argmax(np.squeeze(output))
        result.append(output)
        if output == Char2Num["e"]:
            break
        decInp[:, 0] = output
        step += 1
    decodedResult = [Num2Char[i] for i in result]
    print(decodedResult)

a = 123
b = 2343
Inp = str(a)+"+"+str(b)
Predict(Inp)
print(a+b)