from random import randint
import numpy as np
import tensorflow as tf

class Generate_Data():
    def __init__(self, chars, quantity, max_value):
        self.chars = chars
        self.char_indices =  dict((c, i) for i, c in enumerate(self.chars)) 
        self.indices_chars = dict((i, c) for i, c in enumerate(self.chars))
        data, self.max_inp_len, self.max_out_len = self.generate(quantity, max_value)
        self.inp, self.out = self.encode(data)
    def generate(self, quantity, max_value):
        quantity = int(quantity / 2)
        data = []
        seen = set()
        max_inp_len, max_out_len = 0, 0
        for i in range(1):
            while len(data) < (quantity * (i + 1)):
                string = ""
                string_rev = ""
                a = randint(1, max_value)
                b = randint(1, max_value)
                if i == 0 :
                    string = str(a) + "+" + str(b)
                    string_rev = str(b) + str(a)
                    result = a + b
                elif i == 1 : 
                    string = str(a) + "-" + str(b)
                    result = a - b
                if (string not in list(seen)) and (string_rev not in list(seen)):
                    data.append([string, str(result)])
                    data.append([string_rev, str(result)])
                seen.add(string)
                seen.add(string_rev)
                if max_inp_len < len(string):
                    max_inp_len = len(string)
                if max_out_len <  len(str(result)):
                    max_out_len = len(str(result))
        return data, max_inp_len, max_out_len
    
    def encode(self, data):   
        inp = np.zeros(shape = (len(data), self.max_inp_len, len(self.chars)), dtype = bool)
        out = np.zeros(shape = (len(data), self.max_out_len, len(self.chars)), dtype = bool)
        for i in range(len(data)):
            for j in range(self.max_inp_len):
                try:
                    inp[i, j, self.char_indices[data[i][0][j]]] = 1
                except:
                    inp[i, j, self.char_indices[" "]] = 1
            for j in range(self.max_out_len):
                try:
                    out[i, j, self.char_indices[data[i][1][j]]] = 1
                except:
                    out[i, j, self.char_indices[" "]] = 1
        return inp, out

def generate_model(max_inp_len, max_out_len, char_quantity, lstm_units):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(lstm_units, 
                                   input_shape = (max_inp_len, char_quantity)))
    model.add(tf.keras.layers.RepeatVector(max_out_len))
    model.add(tf.keras.layers.LSTM(lstm_units, return_sequences = True))
    model.add(tf.keras.layers.Dense(char_quantity, activation = "softmax"))
    model.compile(optimizer = "Adam", loss = "categorical_crossentropy", metrics = ["accuracy"])
    return model




if __name__ == "__main__":
    train_data = Generate_Data("0123456789+ ", 100000, 1000)
    model = generate_model(train_data.max_inp_len, train_data.max_out_len, len(train_data.chars), 128)
    model.fit(train_data.inp, train_data.out, epochs = 20, batch_size = 32, validation_split = 0.1)
    test_data = Generate_Data("0123456789+ ", 10000, 1000)
    model.evaluate(test_data.inp, test_data.out)







    

