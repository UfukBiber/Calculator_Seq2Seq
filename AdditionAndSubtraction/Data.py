import numpy as np
import random


Characters = "_se0123456789+-*"

Characters2Numbers = {c:i for i,c in enumerate(list(Characters))}
Numbers2Characters = {i:c for i,c in enumerate(list(Characters))}


Quantity_Of_Training_Data = 50000
Min_Value_Of_Training_Data = 10000
Max_Value_Of_Training_Data = 500000


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
    OutLength = len(str(2 * Max)) + 2
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
            outputSubtraction = str(x-y) +"e"


            if len(EncInpAddition) < InpLength:
                EncInpAddition = EncInpAddition +  "_"*(InpLength-len(EncInpAddition))
            if len(EncInpSubtraction) < InpLength:
                EncInpSubtraction =EncInpSubtraction +  "_"*(InpLength-len(EncInpSubtraction))
            
            if len(outputAddition) < OutLength:
                outputAddition =  outputAddition  + "_"*(OutLength-len(outputAddition))  
            if len(outputSubtraction) < OutLength:
                outputSubtraction = outputSubtraction +  "_"*(OutLength-len(outputSubtraction))  
            
            if len(DecInpAddition) < OutLength:
                DecInpAddition =  DecInpAddition + "_"*(OutLength-len(DecInpAddition))   
            if len(DecInpSubtraction) < OutLength:
                DecInpSubtraction =   DecInpSubtraction + "_"*(OutLength-len(DecInpSubtraction))  
            
            EncInp.append(VectorizeString(EncInpAddition))
            DecInp.append(VectorizeString(DecInpAddition))
            Out.append(VectorizeString(outputAddition))
            EncInp.append(VectorizeString(EncInpSubtraction))
            DecInp.append(VectorizeString(DecInpSubtraction))
            Out.append(VectorizeString(outputSubtraction))

            print("\t Preparing the data : %i/%i"%(len(EncInp), quantity), end="\r")
    print("\n")

    return np.asarray(EncInp, dtype="float32"), np.asarray(DecInp, dtype="float32"), np.asarray(Out, dtype="float32")


def notInclude(data, array):
    if data not in array:
        return True
    return False


if __name__ == "__main__":
    EncInp, DecInp, Out = PrepareData(Quantity_Of_Training_Data, Min_Value_Of_Training_Data ,Max_Value_Of_Training_Data)
    np.save("EncInp", EncInp)
    np.save("DecInp", DecInp)
    np.save("Output", Out)