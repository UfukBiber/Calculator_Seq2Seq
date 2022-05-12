import random
import numpy as np


Characters = "se0123456789+_"
Characters2Numbers = {c:i for i,c in enumerate(list(Characters))}
Numbers2Characters = {i:c for i,c in enumerate(list(Characters))}

def PrepareData(quantity, Min, Max):
    EncInp = []
    DecInp = []
    Out = []
    allData = []
    EncInpLength = len(str(Max)) * 2 + 2
    OutLength = len(str(Max * 2 ))
    while len(EncInp) < quantity:
        x = random.randint(Min, Max)
        y = random.randint(Min, Max)
        if notInclude([x, y], allData):
            allData.append([x, y])
            result = str(x+y)
            dec_Inp = "s" + result 
            output = result + "e"
            if len(output) < OutLength:
                output = output  + "_" * (OutLength-len(output))
                dec_Inp = dec_Inp + "_" * (OutLength-len(dec_Inp))
            x = str(x)
            y = str(y)
            Inp = x + "+" + y + "e"
            if len(Inp) < EncInpLength:
                Inp = Inp + "_" * (EncInpLength - len(Inp))
            EncInp.append(VectorizeString(Inp))
            DecInp.append(VectorizeString(dec_Inp))
            Out.append(VectorizeString(output))
        print("\t Preparing the data : %i/%i"%(len(EncInp), quantity), end="\r")
    print("\n")
    return np.asarray(EncInp), np.asarray(DecInp), np.asarray(Out)

def VectorizeString(string):
    vectorizedString = []
    string = list(string)
    for chr in string:
        vectorizedString.append(Characters2Numbers[chr])
    return vectorizedString

def notInclude(data, array):
    data_2 = [data[1], data[0]]
    if data_2 not in array and data not in array:
        return True
    return False

Inp, decInp, Out = PrepareData(10, 50000, 150000)

for elements in decInp:
    string = ""
    for j in elements:
        string += Numbers2Characters[j]
    print(string)