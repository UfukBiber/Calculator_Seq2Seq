import numpy as np
import tensorflow as tf
import random
Characters = "se+=0123456789"
Characters2Numbers = {c:i for i,c in enumerate(list(Characters))}
Numbers2Characters = {i:c for i,c in enumerate(list(Characters))}




QUANTITY = 5000
VALIDATION_SPLIT = 0.1
TRAIN_LENGTH = int(QUANTITY * (1 - VALIDATION_SPLIT))
MIN = 4
MAX = 7
EMBED_DIMS = 128
DENSE_DIMS = 1024
NUM_HEADS = 8
BATCH_SIZE = 128
SEQ_LENGTH = 18
DropRate = 0.2

InptextVectorization = tf.keras.layers.TextVectorization(
    standardize = None,
    split = "character",
    output_mode = "int",
    output_sequence_length = SEQ_LENGTH,
    vocabulary = list(Characters)
)
TartextVectorization = tf.keras.layers.TextVectorization(
    standardize = None,
    split = "character",
    output_mode = "int",
    output_sequence_length = SEQ_LENGTH+1,
    vocabulary = list(Characters)
)

def PrepareData(Quantity, Max):
    Inp, Tar = [], []
    allData = []
    for i in range(0, Max):
        for j in range(0, Max):
            size = 0
            start_i = 10**i
            end_i =  10**(i+1)
            start_j =  10**j
            end_j = 10**(j+1)
            if i<=2 or j <=2:
                sizeMax = 50
            elif i <=4 or j <= 4:
                sizeMax = 90
            elif i == j:
                sizeMax = Quantity*2
            else:
                sizeMax = Quantity
            while size < sizeMax:
                x = random.randint(start_i, end_i)
                y = random.randint(start_j, end_j)
                if [x, y] not in allData:
                    allData.append([x, y])
                    Inp.append(str(x) + "+" + str(y)+"=")
                    Tar.append("s"+str(x+y)+"e")
                    if x != y:
                        Inp.append(str(y) + "+" + str(x)+"=")
                        Tar.append("s"+str(x+y)+"e")
                    size += 1
                print("\t Preparing the data : %i"%(len(Inp)), end="\r")
    print("\n")
    return Inp, Tar
try:
    with open("TransformerData.txt", "r") as f:
        Inp, Tar = [], []
        for line in f:
            line = line.replace("\n", "")
            inp, tar = line.split("\t")
            Inp.append(inp)
            Tar.append(tar)
        f.close()
    print("Saved Data will be used")
except:
    print("New data will be used")
    Inp, Tar = PrepareData(QUANTITY, MAX)
    with open("TransformerData.txt", "w") as f:
        for i in range(len(Inp)):
            f.write(Inp[i]+"\t"+Tar[i] + "\n")
        f.close()

trainInp, trainTar = Inp[:TRAIN_LENGTH], Tar[:TRAIN_LENGTH]
valInp, valTar = Inp[TRAIN_LENGTH:], Tar[TRAIN_LENGTH:]


random.Random(231).shuffle(Inp)
random.Random(231).shuffle(Tar)


train_ds = tf.data.Dataset.from_tensor_slices((trainInp, trainTar))
train_ds = train_ds.batch(256)
val_ds =  tf.data.Dataset.from_tensor_slices((valInp, valTar))
val_ds = val_ds.batch(256)


def format_data(Inp, Tar):
    Inp = InptextVectorization(Inp)
    Tar = TartextVectorization(Tar)
    return ({"encoder_input":Inp, "decoder_input":Tar[:, :-1]}, Tar[:, 1:])

train_ds = train_ds.map(format_data, num_parallel_calls=4)
val_ds = val_ds.map(format_data, num_parallel_calls=4)


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
    def call(self, inputs, training, mask = None):
        if mask is not None:
            mask = mask[:, tf.newaxis, :]
        attentionOutput = self.attention(inputs, inputs, attention_mask = mask)
        attentionOutput = self.dropout1(attentionOutput, training = training)
        projInput = self.layerNormalization1(inputs + attentionOutput)
        projOut = self.denseProj(projInput)
        projOut = self.dropout2(projOut, training = training)
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

    def call(self, inputs, encoderOut, training, mask = None):
        causal_mask = self.get_causal_attention_mask(inputs)
        if mask is not None:
            mask = tf.cast(mask[:, tf.newaxis, :], dtype = tf.int32)
            mask = tf.minimum(mask, causal_mask)
        attentionOut_1 = self.attention1(query = inputs, value = inputs, key = inputs, attention_mask = causal_mask)
        attentionOut_1 = self.dropout1(attentionOut_1, training = training)
        attentionOut_1 = self.layerNormalization1(inputs + attentionOut_1)
        attentionOut_2 = self.attention2(query = attentionOut_1, value = encoderOut, key = encoderOut, attention_mask = mask)
        attentionOut_2 = self.dropout2(attentionOut_2, training = training)
        attentionOut_2 = self.layerNormalization2(attentionOut_1 + attentionOut_2)
        projOut = self.denseProj(attentionOut_2)
        projOut = self.dropout3(projOut, training = training)
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

lenWords = len(InptextVectorization.get_vocabulary())
encoderInput = tf.keras.layers.Input(shape = (None,), name = "encoder_input")
embeddedEncoder = PositionalEmbedding(lenWords, EMBED_DIMS, SEQ_LENGTH)(encoderInput)
encoderOut = TransformerEncoder(EMBED_DIMS, NUM_HEADS, DENSE_DIMS)(embeddedEncoder)

decoderInput = tf.keras.layers.Input(shape = (None, ), name = "decoder_input")
embeddedDecoder = PositionalEmbedding(lenWords, EMBED_DIMS, SEQ_LENGTH)(decoderInput)
decoderOut = TransformerDecoder(EMBED_DIMS, NUM_HEADS, DENSE_DIMS)(embeddedDecoder, encoderOut)
decoderOut = tf.keras.layers.Dropout(0.3)(decoderOut)
decoderOut = tf.keras.layers.Dense(lenWords, activation = "softmax")(decoderOut)



model = tf.keras.models.Model(inputs = [encoderInput, decoderInput], outputs = decoderOut)
print(model.summary())
try:
    model.load_weights("TransformerModel/Transformer")
    print("Saved model will be used.")
except:
    print("New model will be used.")

model.compile(optimizer = "adam", loss = "sparse_categorical_crossentropy", metrics = ["accuracy"])


model.fit(train_ds, validation_data = val_ds, epochs = 100, callbacks = [tf.keras.callbacks.ModelCheckpoint("TransformerModel/Transformer", save_best_only = True, save_weights_only = True)])
