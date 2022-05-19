import tensorflow as tf
import numpy as np


model = tf.keras.models.load_model("my_model")
Characters = "_se0123456789+-*"

Characters2Numbers = {c:i for i,c in enumerate(list(Characters))}
Numbers2Characters = {i:c for i,c in enumerate(list(Characters))}


x = 111
y = 234

Inp = str(x) + "+" + str(y)
Inp = Inp + "_" * (13 - len(Inp))
VectorInp = np.asarray([[Characters2Numbers[element] for element in list(Inp)]])


output = np.argmax(np.squeeze(model.predict(VectorInp)), axis = -1)

string = "".join([Numbers2Characters[i] for i in output])

print(string)