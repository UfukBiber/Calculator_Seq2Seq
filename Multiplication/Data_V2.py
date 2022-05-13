import numpy as np
import random


Characters = "se0123456789+-*_"

Characters2Numbers = {c:i for i,c in enumerate(list(Characters))}
Numbers2Characters = {i:c for i,c in enumerate(list(Characters))}




def VectorizeString(string):
    vectorizedString = []
    string = list(string)
    for chr in string:
        vectorizedString.append(Characters2Numbers[chr])
    return vectorizedString



def PrepareData(quantity, Min, Max):
    EncInp = []
    DecInp = []
    Out = []
    allData = []
    OutLength = len(str((Max-1) * (Max-1))) + 1
    InpLength = len(str(Max)) * 2 + 1
    while len(EncInp)  < quantity:
        x = random.randint(Min, Max)
        y = random.randint(Min, Max)
        if notInclude([x, y], allData):
            allData.append([x, y])
            EncInpAddition = str(x) + "+" + str(y)
            DecInpAddition = "s" + str(x+y)
            outputAddition = str(x+y) + "e"
            EncInpSubtraction = str(x) + "-" + str(y)
            DecInpSubtraction = "s" + str(x-y)
            outputSubtraction = str(x-y) + "e"
            EncInpMultiplication= str(x) + "*" + str(y)
            DecInpMultiplication = "s" + str(x*y)
            outputMultiplication = str(x*y) + "e"

            if len(EncInpAddition) < InpLength:
                EncInpAddition = EncInpAddition +  "_"*(InpLength-len(EncInpAddition))
            if len(EncInpSubtraction) < InpLength:
                EncInpSubtraction =EncInpSubtraction +  "_"*(InpLength-len(EncInpSubtraction))
            if len(EncInpMultiplication) < InpLength:
                EncInpMultiplication =EncInpMultiplication +  "_"*(InpLength-len(EncInpMultiplication))
            if len(outputAddition) < OutLength:
                outputAddition =outputAddition +  "_"*(OutLength-len(outputAddition)) 
            if len(outputSubtraction) < OutLength:
                outputSubtraction = outputSubtraction + "_"*(OutLength-len(outputSubtraction))  
            if len(outputMultiplication) < OutLength:
                outputMultiplication =outputMultiplication +  "_"*(OutLength-len(outputMultiplication)) 
            if len(DecInpAddition) < OutLength:
                DecInpAddition = DecInpAddition  + "_"*(OutLength-len(DecInpAddition)) 
            if len(DecInpSubtraction) < OutLength:
                DecInpSubtraction = DecInpSubtraction + "_"*(OutLength-len(DecInpSubtraction)) 
            if len(DecInpMultiplication) < OutLength:
                DecInpMultiplication = DecInpMultiplication+ "_"*(OutLength-len(DecInpMultiplication)) 
            
            EncInp.append(VectorizeString(EncInpAddition))
            DecInp.append(VectorizeString(DecInpAddition))
            Out.append(VectorizeString(outputAddition))
            EncInp.append(VectorizeString(EncInpSubtraction))
            DecInp.append(VectorizeString(DecInpSubtraction))
            Out.append(VectorizeString(outputSubtraction))
            EncInp.append(VectorizeString(EncInpMultiplication))
            DecInp.append(VectorizeString(DecInpMultiplication))
            Out.append(VectorizeString(outputMultiplication))
            print("\t Preparing the data : %i/%i"%(len(EncInp), quantity), end="\r")
    print("\n")

    return np.asarray(EncInp, dtype="float32"), np.asarray(DecInp, dtype="float32"), np.asarray(Out, dtype="float32")


def notInclude(data, array):
    if data not in array:
        return True
    return False



# EncInp, DecInp, Out = PrepareData(1000, 10 ,5000)

# print(EncInp.shape)
# print(DecInp.shape)
# print(Out.shape)
# # for i in range(len(EncInp)):
# #     print("Enc: ", len(EncInp[i]))
# #     print("Dec: ", len(DecInp[i]))
# #     print("Out: ", len(Out[i]))
