import numpy as np
import tensorflow as tf
import random
Characters = " se+=0123456789"
Characters2Numbers = {c:i for i,c in enumerate(list(Characters))}
Numbers2Characters = {i:c for i,c in enumerate(list(Characters))}



QUANTITY = 20000
VALIDATION_SPLIT = 0.1
TRAIN_LENGTH = int(QUANTITY * (1 - VALIDATION_SPLIT))
MIN = 4
MAX = 7
EMBED_DIMS = 128
DENSE_DIMS = 1024
NUM_HEADS = 8
BATCH_SIZE = 128
SEQ_LENGTH = 18
DropRate = 0.3

def PrepareData(Quantity, Max):
    data = []
    allData = []
    for i in range(0, Max):
        for j in range(0, Max):
            size = 0
            start_i = 10**i
            end_i =  10**(i+1)
            start_j =  10**j
            end_j = 10**(j+1)
            if i<=2 or j <=2:
                sizeMax = 10
            elif i <=4 or j <= 4:
                sizeMax = 30
            elif i == j:
                sizeMax = Quantity*2
            else:
                sizeMax = Quantity
            while size < sizeMax:
                x = random.randint(start_i, end_i)
                y = random.randint(start_j, end_j)
                if [x, y] not in allData:
                    allData.append([x, y])
                    data.append(({"encoder_input":str(x) + "+" + str(y)+"=", "decoder_input":"s"+str(x+y)}, str(x+y)+"e"))
                    data.append(({"encoder_input":str(y) + "+" + str(x)+"=", "decoder_input":"s"+str(x+y)}, str(x+y)+"e"))
                    size += 1
                print("\t Preparing the data : %i"%(len(data)), end="\r")
    print("\n")
    return data
try:
    with open("TransformerData.txt", "r") as f:
        data = []
        for line in f:
            line = line.replace("\n", "")
            encInp, decInp, Out = line.split("\t")
            data.append(({"encoder_input": encInp, "decoder_input": decInp}, Out))
        f.close()
    print("Saved Data will be used")
except:
    print("New data will be used")
    data = PrepareData(QUANTITY, MAX)
    with open("TransformerData.txt", "w") as f:
        for line in data:
            f.write(line[0]["encoder_input"]+"\t"+line[0]["decoder_input"]+"\t"+line[1]+"\n")
        f.close()
    

random.seed(23415)
random.shuffle(data)

def VectorizeData(data, max_len = None):
    vectorizedData = []
    for element in data:
        newElement = []
        for char in list(element):
            newElement.append(Characters2Numbers[char])
        vectorizedData.append(newElement)
    vectorizedData = tf.keras.preprocessing.sequence.pad_sequences(vectorizedData, padding="post", maxlen = max_len)
    return vectorizedData


def format_data(pairs, seqLength):
    Inp = {"encoder_input":[], "decoder_input":[]}
    Out = []
    for pair in pairs:
        Inp["encoder_input"].append(pair[0]["encoder_input"])
        Inp["decoder_input"].append(pair[0]["decoder_input"])
        Out.append(pair[1])
    Inp["encoder_input"] = VectorizeData(Inp["encoder_input"], SEQ_LENGTH)
    Inp["decoder_input"] = VectorizeData(Inp["decoder_input"], SEQ_LENGTH)
    Out = VectorizeData(Out, SEQ_LENGTH)
    return Inp, Out


trainData = data[:TRAIN_LENGTH]
valData = data[TRAIN_LENGTH:]

trainInp, trainOut = format_data(trainData)
valInp, valOut = format_data(valData)

INP_SEQ_LENGTH = trainInp["encoder_input"].shape[-1]
OUT_SEQ_LENGTH = trainOut.shape[-1]

train_ds = tf.data.Dataset.from_tensor_slices((trainInp, trainOut)).batch(BATCH_SIZE)
val_ds = tf.data.Dataset.from_tensor_slices((valInp, valOut)).batch(BATCH_SIZE)



train_ds = train_ds.shuffle(1024).prefetch(1024)
val_ds = val_ds.shuffle(1024).prefetch(1024)

class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, maxWord, embeddingDims, seqLength, **kwargs):
        super().__init__(**kwargs)
        self.maxWord = maxWord
        self.embeddingDims = embeddingDims
        self.seqLength = seqLength
        self.embeddingWord = tf.keras.layers.Embedding(maxWord, embeddingDims)
        self.embeddingPosition = tf.keras.layers.Embedding(seqLength, embeddingDims)
    
    def call(self, inputs):
        length = tf.shape(inputs)[-1]
        position = tf.range(start = 0, limit = length, delta = 1)
        embeddedTokens = self.embeddingWord(inputs)
        embeddedPos = self.embeddingPosition(position)
        return embeddedTokens + embeddedPos
    
    def compute_mask(self, inputs, mask = None):
        return tf.math.not_equal(inputs, 0)

    def get_config(self):
        baseConfig = super().get_config()
        return {**baseConfig, 
                "maxWord":self.maxWord, 
                "embeddingDims":self.embeddingDims,
                "seqLength":self.seqLength}
    
class TransformerEncoder(tf.keras.layers.Layer):
    def __init__(self, embedDims, numHeads, denseDims, **kwargs):
        super().__init__(**kwargs)

        self.embedDims = embedDims
        self.numHeads = numHeads
        self.denseDims = denseDims

        self.attention = tf.keras.layers.MultiHeadAttention(numHeads, embedDims)
        self.denseProj = tf.keras.Sequential([
            tf.keras.layers.Dense(denseDims, activation = "relu"),
            tf.keras.layers.Dense(embedDims)
        ])
        self.layerNormalization1 = tf.keras.layers.LayerNormalization() 
        self.layerNormalization2 = tf.keras.layers.LayerNormalization()

        self.dropout1 = tf.keras.layers.Dropout(DropRate)
        self.dropout2 = tf.keras.layers.Dropout(DropRate)

        self.supports_masking = True
    def call(self, inputs, mask = None):
        if mask is not None:
            mask = mask[:, tf.newaxis, :]
        attentionOutput = self.attention(inputs, inputs, attention_mask = mask)
        attentionOutput = self.dropout1(attentionOutput)
        projInput = self.layerNormalization1(inputs + attentionOutput)
        projOut = self.denseProj(projInput)
        projOut = self.dropout2(projOut)
        return self.layerNormalization2(projInput + projOut)
        

    def get_config(self):
        baseConfig = super().get_config()
        return {**baseConfig, 
            "embedDims":self.embedDims,
            "numHeads":self.numHeads,
            "denseDims":self.denseDims}
    
class TransformerDecoder(tf.keras.layers.Layer):
    def __init__(self, embedDims, numHeads, denseDims, **kwargs):
        super().__init__(**kwargs)

        self.embedDims = embedDims
        self.numHeads = numHeads
        self.denseDims = denseDims

        self.attention1 = tf.keras.layers.MultiHeadAttention(numHeads, embedDims)
        self.attention2 = tf.keras.layers.MultiHeadAttention(numHeads, embedDims)
        self.denseProj = tf.keras.Sequential([
            tf.keras.layers.Dense(denseDims, activation = "relu"),
            tf.keras.layers.Dense(embedDims)
        ])
        self.layerNormalization1 = tf.keras.layers.LayerNormalization() 
        self.layerNormalization2 = tf.keras.layers.LayerNormalization() 
        self.layerNormalization3 = tf.keras.layers.LayerNormalization()

        self.dropout1 = tf.keras.layers.Dropout(DropRate)
        self.dropout2 = tf.keras.layers.Dropout(DropRate)
        self.dropout3 = tf.keras.layers.Dropout(DropRate)

        self.supports_masking = True

    def call(self, inputs, encoderOut, mask = None):
        causal_mask = self.get_causal_attention_mask(inputs)
        if mask is not None:
            mask = tf.cast(mask[:, tf.newaxis, :], dtype = tf.int32)
            mask = tf.minimum(mask, causal_mask)
        attentionOut_1 = self.attention1(query = inputs, value = inputs, key = inputs, attention_mask = causal_mask)
        attentionOut_1 = self.dropout1(attentionOut_1)
        attentionOut_1 = self.layerNormalization1(inputs + attentionOut_1)
        attentionOut_2 = self.attention2(query = attentionOut_1, value = encoderOut, key = encoderOut, attention_mask = mask)
        attentionOut_2 = self.dropout2(attentionOut_2)
        attentionOut_2 = self.layerNormalization2(attentionOut_1 + attentionOut_2)
        projOut = self.denseProj(attentionOut_2)
        projOut = self.dropout3(projOut)
        return self.layerNormalization3(attentionOut_2 + projOut)
    
    def get_causal_attention_mask(self, inputs):
        InputShape = tf.shape(inputs)
        i = tf.range(InputShape[1])[:, tf.newaxis]
        j = tf.range(InputShape[1])
        mask = tf.cast(i >= j, dtype = tf.int32)
        mask = tf.reshape(mask, (1, InputShape[1], InputShape[1]))
        mult = tf.concat([tf.expand_dims(InputShape[0], -1), 
                          tf.constant([1, 1], dtype = tf.int32)], axis = 0)
        
        return tf.tile(mask, mult)

    def get_config(self):
        baseConfig = super().get_config()
        return {**baseConfig, 
            "embedDims":self.embedDims,
            "numHeads":self.numHeads,
            "denseDims":self.denseDims}

encoderInput = tf.keras.layers.Input(shape = (None,), name = "encoder_input")
embeddedEncoder = PositionalEmbedding(len(Characters), EMBED_DIMS, INP_SEQ_LENGTH)(encoderInput)
encoderOut = TransformerEncoder(EMBED_DIMS, NUM_HEADS, DENSE_DIMS)(embeddedEncoder)

decoderInput = tf.keras.layers.Input(shape = (None, ), name = "decoder_input")
embeddedDecoder = PositionalEmbedding(len(Characters), EMBED_DIMS, OUT_SEQ_LENGTH)(decoderInput)
decoderOut = TransformerDecoder(EMBED_DIMS, NUM_HEADS, DENSE_DIMS)(embeddedDecoder, encoderOut)
decoderOut = tf.keras.layers.Dropout(0.3)(decoderOut)
decoderOut = tf.keras.layers.Dense(len(Characters), activation = "softmax")(decoderOut)



model = tf.keras.models.Model(inputs = [encoderInput, decoderInput], outputs = decoderOut)
print(model.summary())
try:
    model.load_weights("TransformerModel/Transformer")
    print("Saved model will be used.")
except:
    print("New model will be used.")

model.compile(optimizer = "adam", loss = "sparse_categorical_crossentropy", metrics = ["accuracy"])


model.fit(train_ds, validation_data = val_ds, epochs = 100, callbacks = [tf.keras.callbacks.ModelCheckpoint("TransformerModel/Transformer", save_best_only = True, save_weights_only = True),
                                                                        tf.keras.callbacks.EarlyStopping(monitor = "val_accuracy", patience = 3)])
