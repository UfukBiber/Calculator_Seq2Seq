import numpy as np
import tensorflow as tf
import random
Characters = " se+=0123456789"
Characters2Numbers = {c:i for i,c in enumerate(list(Characters))}
Numbers2Characters = {i:c for i,c in enumerate(list(Characters))}



QUANTITY = 216000
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

try:
    with open("data.txt", "r") as f:
        data = []
        for line in f:
            line = line.replace("\n", "")
            encInp, decInp, Out = line.split("\t")
            data.append(({"encoder_input": encInp, "decoder_input": decInp}, Out))
        f.close()
    print("Saved Data will be used")
except:
    print("New data will be used")
    data = PrepareData(QUANTITY, MIN, MAX)
    with open("data.txt", "w") as f:
        for line in data:
            f.write(line[0]["encoder_input"]+"\t"+line[0]["decoder_input"]+"\t"+line[1]+"\n")
        f.close()
    

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


train_ds = tf.data.Dataset.from_tensor_slices((trainInp, trainOut)).batch(BATCH_SIZE)
val_ds = tf.data.Dataset.from_tensor_slices((valInp, valOut)).batch(BATCH_SIZE)



train_ds = train_ds.shuffle(1024).prefetch(1024)
val_ds = val_ds.shuffle(1024).prefetch(1024)

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

try:
    model.load_weights("EncoderDecoderModel/EncoderDecoder")
    print("Saved model will be used.")
except:
    print("New model will be used.")

model.compile(optimizer = "adam", loss = "sparse_categorical_crossentropy", metrics = ["accuracy"])



model.fit(train_ds, validation_data = val_ds, epochs = 3, callbacks = [tf.keras.callbacks.ModelCheckpoint("EncoderDecoderModel/EncoderDecoder", save_best_only = True, save_weights_only = True),
                                                                        tf.keras.callbacks.EarlyStopping(monitor = "val_accuracy", patience = 3)])
