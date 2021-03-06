import tensorflow as tf
import random
import numpy as np

Characters = " se+=0123456789"
Characters2Numbers = {c:i for i,c in enumerate(list(Characters))}
Numbers2Characters = {i:c for i,c in enumerate(list(Characters))}


QUANTITY = 18
VALIDATION_SPLIT = 0.1
LSTM_UNITS = 256
TRAIN_LENGTH = int(QUANTITY * (1 - VALIDATION_SPLIT))
MIN = 4
MAX = 7
EMBED_DIMS = 32
BATCH_SIZE = 32


def PrepareData(Quantity, Min, Max):
    assert Quantity % 18 == 0, "Quantity must be divisible with 18."
    data = []
    allData = []
    for i in range(Min-1, Max-1):
        for j in range(Min-1, Max-1):
            size = 0
            start_i = 10**i
            end_i =  10**(i+1) -1
            start_j =  10**j
            end_j = 10**(j+1) -1
            if i == j:
                sizeMax = Quantity/9
            else:
                sizeMax = Quantity/18
            while size < sizeMax:
                x = random.randint(start_i, end_i)
                y = random.randint(start_j, end_j)
                if [x, y] not in allData:
                    allData.append([x, y])
                    data.append(({"encoder_input":str(x) + "+" + str(y)+"=", "decoder_input":"s" +str(x+y)}, str(x+y) + "e"))
                    data.append(({"encoder_input":str(y) + "+" + str(x)+"=", "decoder_input":"s" +str(x+y)}, str(x+y) + "e"))
                    size += 1
                print("\t Preparing the data : %i/%i"%(len(data), Quantity), end="\r")
    print("\n")
    return data


data = PrepareData(QUANTITY, MIN, MAX)
random.seed(23415)
random.shuffle(data)

def VectorizeData(data):
    vectorizedData = []
    for element in data:
        newElement = []
        for char in list(element):
            newElement.append(Characters2Numbers[char])
        vectorizedData.append(newElement)
    vectorizedData = tf.keras.preprocessing.sequence.pad_sequences(vectorizedData, padding="post")
    return vectorizedData

def format_data(pairs):
    Inp = {"encoder_input":[], "decoder_input":[]}
    Out = []
    for pair in pairs:
        Inp["encoder_input"].append(pair[0]["encoder_input"])
        Inp["decoder_input"].append(pair[0]["decoder_input"])
        Out.append(pair[1])
    Inp["encoder_input"] = VectorizeData(Inp["encoder_input"])
    Inp["decoder_input"] = VectorizeData(Inp["decoder_input"])
    Out = VectorizeData(Out)
    return Inp, Out


trainData = data[:TRAIN_LENGTH]
valData = data[TRAIN_LENGTH:]

trainInp, trainOut = format_data(trainData)
valInp, valOut = format_data(valData)

EncInp = tf.keras.layers.Input(shape= (None,), name = "encoder_input")
encoderEmbedded = tf.keras.layers.Embedding(len(Characters), EMBED_DIMS, mask_zero = True)(EncInp)
encOut, Enc_State_for, Enc_state_back = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(LSTM_UNITS, return_state = True, name = "GRU_Encoder"))(encoderEmbedded)
states = tf.keras.layers.Concatenate()([Enc_State_for, Enc_state_back])

DecInp = tf.keras.layers.Input(shape=(None,), name = "decoder_input")
decoderEmbedding = tf.keras.layers.Embedding(len(Characters), EMBED_DIMS, mask_zero = True)(DecInp)
decOut, _ = tf.keras.layers.GRU(LSTM_UNITS*2, return_sequences = True, return_state = True, name = "GRU_Encoder")(decoderEmbedding, initial_state = states)
decOut = tf.keras.layers.Dropout(0.5)(decOut)
decOut = tf.keras.layers.Dense(len(Characters), activation = "softmax")(decOut)



model = tf.keras.models.Model(inputs = [EncInp, DecInp], outputs = decOut)
####################
model.load_weights("EncoderDecoderModel/EncoderDecoder")



def Predict(Inp):
    if len(Inp) < 14:
        Inp += " "*(14-len(Inp))
    VectorInp = tf.expand_dims(tf.constant([Characters2Numbers[i] for i in Inp]), axis = 0)
    decInp = np.zeros((1, trainOut.shape[-1]))
    decInp[0, 0] = Characters2Numbers["s"]
    result = []
    step = 0
    while step < trainOut.shape[-1]:
        output = model([VectorInp, decInp])
        output = np.argmax(output[0, step, :])
        print(Numbers2Characters[output])
        result.append(output)
        if "e" == Numbers2Characters[output]:
            break
        else:
            step += 1
            decInp[:, step] = output
        
a = 42
b = 812
Inp = str(a)+"+"+str(b)+"="
Predict(Inp)
print("\n")
print(a+b)