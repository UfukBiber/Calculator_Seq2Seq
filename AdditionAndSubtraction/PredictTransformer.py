import tensorflow as tf
import random
import numpy as np




Characters = " se+=0123456789"
Characters2Numbers = {c:i for i,c in enumerate(list(Characters))}
Numbers2Characters = {i:c for i,c in enumerate(list(Characters))}


QUANTITY = 10000
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


encoderInput = tf.keras.layers.Input(shape = (None,), name = "encoder_input")
embeddedEncoder = PositionalEmbedding(len(Characters), EMBED_DIMS, SEQ_LENGTH)(encoderInput)
encoderOut = TransformerEncoder(EMBED_DIMS, NUM_HEADS, DENSE_DIMS)(embeddedEncoder)

decoderInput = tf.keras.layers.Input(shape = (None, ), name = "decoder_input")
embeddedDecoder = PositionalEmbedding(len(Characters), EMBED_DIMS, SEQ_LENGTH)(decoderInput)
decoderOut = TransformerDecoder(EMBED_DIMS, NUM_HEADS, DENSE_DIMS)(embeddedDecoder, encoderOut)
decoderOut = tf.keras.layers.Dropout(0.3)(decoderOut)
decoderOut = tf.keras.layers.Dense(len(Characters), activation = "softmax")(decoderOut)


model = tf.keras.models.Model(inputs = [encoderInput, decoderInput], outputs = decoderOut)
print(model.summary())

model = tf.keras.models.Model(inputs = [encoderInput, decoderInput], outputs = decoderOut)
####################
model.load_weights("TransformerModel/Transformer")



def Predict(Inp):
    if len(Inp) < SEQ_LENGTH:
        Inp += " "*(SEQ_LENGTH-len(Inp))
    VectorInp = tf.expand_dims(tf.constant([Characters2Numbers[i] for i in Inp]), axis = 0)
    decInp = np.zeros((1, SEQ_LENGTH))
    decInp[0, 0] = Characters2Numbers["s"]
    result = []
    step = 0
    while step < SEQ_LENGTH:
        output = model([VectorInp, decInp])
        output = np.argmax(output[0, step, :])
        result.append(Numbers2Characters[output])
        if "e" == Numbers2Characters[output]:
            break
        else:
            step += 1
            decInp[:, step] = output
    print("".join(result))  
a = 79
b = 16
Inp = str(a)+"+"+str(b)+"="
Predict(Inp)
print("\n")
print(a+b)