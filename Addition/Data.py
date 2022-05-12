import numpy as np
import random



Characters = "0123456789+_"
Characters2Numbers = {c:i for i,c in enumerate(list(Characters))}
Numbers2Characters = {i:c for i,c in enumerate(list(Characters))}

def PrepareData(quantity, Min, Max):
    Inp = []
    Out = []
    allData = []
    InpLength = len(str(Max)) * 2 + 1
    OutLength = len(str(Max * 2 - 2))
    while len(Inp) < quantity:
        x = random.randint(Min, Max)
        y = random.randint(Min, Max)
        if notInclude([x, y], allData):
            allData.append([x, y])
            output = str(x+y)
            if len(output) < OutLength:
                output = output + "_"*(OutLength-len(output))  
            Input = str(x) + "+" + str(y)
            if len(Input) < InpLength:
                Input = Input + "_"*(InpLength-len(Input)) 
            Inp.append(VectorizeString(Input))
            Out.append(VectorizeString(output))
        print("\t Preparing the data : %i/%i"%(len(Inp), quantity), end="\r")
    print("\n")
    return np.asarray(Inp), np.asarray(Out)
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

Inp, Out = PrepareData(10, 50000, 150000)



for elements in Out:
    string = ""
    for j in elements:
        string += Numbers2Characters[j]
    print(string)