#!/usr/bin/python

import numpy as np
import math
import time
import pickle as cPickle
import fix_yahoo_finance as yf 


####################################################################################
#
# This section contains miscellaneous functions used in the program.
#
####################################################################################


# save() function to save the trained network to a file
def save(ann, file_name):
    with open(file_name, 'wb') as fp:
        cPickle.dump(ann, fp)

# restore() function to restore the file
def load(file_name):
    with open(file_name, 'rb') as fp:
        nn = cPickle.load(fp)
    return nn

# This function is used to facilitate file navigation and saving
# For the integers 1-3, it converts the int to the string
# equivalent.
def intToString(num):
    if num == 1:
        return 'one'
    elif num == 2:
        return 'two'
    else:
        return 'three'


def isAnn(netStyle):
    if netStyle == 'ann':
        return True
    else:
        return False


####################################################################################
#
# This section contains the functions to prepare the raw stock data for prediction.
#
####################################################################################


def setPredictionData(array,inputNodes,netStyle):
    predictionData=np.asarray(normalizeArray2(array)[-inputNodes:])
    if netStyle=='ann':
        predictionData=predictionData.reshape([-1,inputNodes,1])
    else:
        size = int(math.sqrt(inputNodes))
        predictionData=predictionData.reshape([-1,size,size,1])
    return predictionData  


####################################################################################
#
# This section contains the functions to prepare the raw stock data for testing.
#
####################################################################################


# Retrieve stock data from Yahoo Finance. Supply the stock's ticker
# Along with the start and end dates of interest.
def getStock(ticker,start,end):
  dataRaw = yf.download(ticker,start,end)
  closeValues=dataRaw.Close.values
  openValues=dataRaw.Open.values
  change, percentChange=getChange(openValues,closeValues)
  return (openValues, closeValues,change, percentChange)


# Normalize each array from the stock data
def normalizeInputData(data):
    for array in range(0,len(data)):
        data[array]=normalizeArray2(data[array])
    return data

# Normalize an array while ensuring values maintain their sign
def normalizeArray2(array):
    scale=max(max(array),-min(array))
    for val in range(0,len(array)):
        array[val]=array[val]/scale
    return array

# Return the change and percent change bfor a stock between close and open
def getChange(open,close):
    change=[]
    percentChange=[]
    for day in range(0,len(open)):
        change.append(close[day]-open[day])
        percentChange.append(change[day]/open[day])
    return change,percentChange


def getInputData(stock,length,reshape):
    train_d,train_t,test_d,test_t=setInputData(stock[2],length,binary=True)
    train_d=normalizeInputData(train_d)
    test_d=normalizeInputData(test_d)
    return reshapeArrays(train_d,train_t,test_d,test_t, reshape,length)


# Create the data arrays for input to TFLearn. In order to do this,
# The algorithm begins at the first day and saves the next "inputNodes"
# of values for the input data. It then sets the "inputNodes"+1 value
# for the corresponding target data. This is repeated for the first 70%
# of the data source. After which, the process is repeated for the 
# validation data.
def setInputData(array,inputNodes,binary=False):

 # Initialize arrays and needed values
    testD=[]
    testT=[]
    trainD=[]
    trainT=[]
    trainToTestRation=.7
    partition=int(trainToTestRation*(len(array)-inputNodes))

 # Create the training data arrays using the first partition of data
    for i in range(0,partition):
        trainD.append(array[i:inputNodes+i])
        if binary:
            trainT.append(getBinaryTarget2(array[i+inputNodes-1],array[i+inputNodes]))
        else:
            trainT.append(array[i+inputNodes])

 # Create the training data using the second partition of data
    for i in range(partition,len(array)-inputNodes):
        testD.append(array[i:inputNodes+i])
        if binary:
            testT.append(getBinaryTarget2(array[i+inputNodes-1],array[i+inputNodes]))
        else:
            testT.append(array[i+inputNodes])

 # Convert the data arrays to np arrays
    trainD=np.asarray(trainD)
    trainT=np.asarray(trainT)
    testD=np.asarray(testD)
    testT=np.asarray(testT)
    return (trainD, trainT, testD, testT)

# This function creates the target datum
# A buy is represented by [1,0]
# A sell is represented by [0,1]
def getBinaryTarget2(today,tomorrow):
    if tomorrow> 0:
        return [1,0]
    else:
        return [0,1]


def reshapeArrays(train_d,train_t,test_d,test_t,reshape, length):
    if reshape:
        train_d=train_d.reshape([-1,int(math.sqrt(length)),int(math.sqrt(length)),1])
        test_d=test_d.reshape([-1,int(math.sqrt(length)),int(math.sqrt(length)),1])
    else:
       train_d=train_d.reshape([-1,length,1])
       test_d=test_d.reshape([-1,length,1])

    train_t=train_t.reshape([-1,2])
    test_t=test_t.reshape([-1,2])
    return train_d,train_t,test_d,test_t


####################################################################################
#
# This section contains the functions to save the parameters of the best network
# found during mass testing. The functions create a text file, to allow the 
# user to readily view the results. A tuple is pickled with the same name in the 
# same location to facilitate reloading the best network.
#
####################################################################################


def writeFile1Level(leng,epoch,result,lr,func,depth,MBS,saveName,saveBase):
        file=open("results/ann/oneLevel/"+saveName+".txt","w")   
        line='Accuracy: '+`result` + '\nNodes: '+`depth` +'\nEpoch: '+`epoch`+'\nActivation: '+func+'\nMBS: '+`MBS`+'\nLength: '+`leng`+'\n Learning Rate: '+`lr`
        file.write(line)
        save(('ann',leng,lr,[func],[depth]),"results/ann/oneLevel/"+saveName+".pck")
        file.close()


def writeFile2Level(leng,epoch,result,lr,func,func1, depth,depth1, MBS,saveName,saveBase):
        file=open("results/ann/twoLevel/"+saveName+".txt","w")   
        line='Accuracy: '+`result`
        line=line + '\nNodes: '+`depth`+'\nNodes 2: '+`depth1`
        line=line + '\nEpoch: '+`epoch`
        line=line + '\nActivation: '+func+'\nActivation 2: '+func1
        line=line + '\nMBS: '+`MBS`+'\nLength: '+`leng`+ '\nLearning Rate: '+`lr`
        file.write(line)
        save(('ann',leng,lr,[func,func1],[depth,depth1]),"results/ann/twoLevel/"+saveName+".pck")
        file.close()


def writeFile3Level(leng,epoch,result,lr,func,func1,func2, depth,depth1,depth2, MBS,saveName,saveBase):
        file=open("results/ann/threeLevel/"+saveName+".txt","w")
        line='Accuracy: '+`result` 
        line=line + '\nNodes: '+`depth`+'\nNodes 2: '+`depth1` +'\nNodes 3: '+`depth2`
        line=line + '\nEpoch: '+`epoch`
        line=line + '\nActivation: '+func+'\nActivation 2: '+func1+ '\nActivation 3: '+func2
        line=line + '\nMBS: '+`MBS`+'\nLength: '+`leng`+ '\nLearning Rate: '+`lr`
        file.write(line)
        save(('ann',leng,lr,[func,func1,func2],[depth,depth1,depth2]),"results/ann/threeLevel/"+saveName+".pck")
        file.close()


def writeConv1Level(leng,epoch,result,lr,func,depth,MBS,saveName,saveBase):
        file=open("results/conv/oneLevel/"+saveName+".txt","w")    
        line='Accuracy: '+`result` + '\nNodes: '+`depth` +'\nEpoch: '+`epoch`+'\nActivation: '+func+'\nMBS: '+`MBS`+'\nLength: '+`leng`+'\nLearning Rate: '+`lr`
        file.write(line)
        save(('conv',leng,lr,[func],[depth]),"results/conv/oneLevel/"+saveName+".pck")
        file.close()

    

def writeConv2Level(leng,epoch,result,lr,func,func1, depth,depth1, MBS,saveName,saveBase):
        file=open("results/conv/twoLevel/"+saveName+".txt","w")   
        line='Accuracy: '+`result`
        line=line + '\nNodes: '+`depth`+'\nNodes 2: '+`depth1`
        line=line + '\nEpoch: '+`epoch`
        line=line + '\nActivation: '+func+'\nActivation 2: '+func1
        line=line + '\nMBS: '+`MBS`+'\nLength: '+`leng`+ '\nLearning Rate: '+`lr`
        file.write(line)
        save(('conv',leng,lr,[func,func1],[depth,depth1]),"results/conv/twoLevel/"+saveName+".pck")
        file.close()

       
def writeConv3Level(leng,epoch,result,lr,func,func1,func2, depth,depth1,depth2, MBS,saveName,saveBase):
        file=open("results/conv/threeLevel/"+saveName+".txt","w")   
        line='Accuracy: '+`result`
        line=line + '\nNodes: '+`depth`+'\nNodes 2: '+`depth1`+'\nNodes 3: '+`depth2`
        line=line + '\nEpoch: '+`epoch`
        line=line + '\nActivation: '+func+'\nActivation 2: '+func1+'\nActivation 3: '+func2
        line=line + '\nMBS: '+`MBS`+'\nLength: '+`leng`+ '\nLearning Rate: '+`lr`
        file.write(line)
        save(('conv',leng,lr,[func,func1,func2],[depth,depth1,depth2]),"results/conv/threeLevel/"+saveName+".pck")
        file.close()


def chooseFileWriter(memory,netStyle,epoch,result,lr,funcs, depths, MBS,saveName,saveBase):
    numLayers=len(funcs)
    if netStyle == 'ann':
        if numLayers == 1:
            writeFile1Level(memory,epoch,result,lr,funcs[0],depths[0],MBS,saveName,saveBase)
        elif numLayers == 2:
            writeFile2Level(memory,epoch,result,lr,funcs[0],funcs[1],depths[0], depths[1],MBS,saveName,saveBase)
        else:
            writeFile3Level(memory,epoch,result,lr,funcs[0],funcs[1],funcs[2],depths[0], depths[1],depths[2],MBS,saveName,saveBase)
    else:
        if numLayers == 1:
            writeConv1Level(memory,epoch,result,lr,funcs[0],depths[0],MBS,saveName,saveBase)
        elif numLayers == 2:
            writeConv2Level(memory,epoch,result,lr,funcs[0],funcs[1],depths[0], depths[1],MBS,saveName,saveBase)
        else:
            writeConv3Level(memory,epoch,result,lr,funcs[0],funcs[1],funcs[2],depths[0], depths[1],depths[2],MBS,saveName,saveBase)


####################################################################################
#
# This section contains the functions to print the progress during mass testing.
#
####################################################################################

# Print the progress of the looping functions
def printProgress3(startTime,count,totalRuns,lr,func,func1,func2,depth,depth1,depth2,mbs,leng,result,bestResult):
                    currentTime=time.time()-startTime
                    print "========================================================\n"
                    print "Best Result: %f" %(bestResult)
                    print "Previous Result: %f\n\n" %(result)
                    print "This is run %i/%i" %(count,totalRuns)
                    print "Time elapsed: %f/%f minutes" %(round(currentTime/60,2),round(currentTime*totalRuns/count/60,2))
                    print "Time elapsed: %f/%f hours" %(round(currentTime/3600,2),round(currentTime*totalRuns/count/3600,2))
                    print "lr: %f \nActivation Functions: %s\t%s\t%s \nNodes: %i\t%i\t%i \nMBS: %i \nLength: %i" %(lr,func,func1,func2, depth,depth1,depth2, mbs,leng)
                    print "=======================================================\n"


# Print the progress of the looping functions
def printProgress2(startTime,count,totalRuns,lr,func,func1,depth,depth1,mbs,leng,result, bestResult):
                    currentTime=time.time()-startTime
                    print "========================================================\n"
                    print "Best Result: %f" %(bestResult)
                    print "Previous Result: %f\n\n" %(result)
                    print "This is run %i/%i" %(count,totalRuns)
                    print "Time elapsed: %f/%f minutes" %(round(currentTime/60,2),round(currentTime*totalRuns/count/60,2))
                    print "Time elapsed: %f/%f hours" %(round(currentTime/3600,2),round(currentTime*totalRuns/count/3600,2))
                    print "lr: %f \nActivation Functions: %s\t%s \nNodes: %i\t%i \nMBS: %i \nLength: %i" %(lr,func,func1, depth,depth1, mbs,leng)
                    print "=======================================================\n"


# Print the progress of the looping functions
def printProgress(startTime,count,totalRuns,lr,func,depth,mbs,leng,result,bestResult):
                    currentTime=time.time()-startTime
                    print "========================================================\n"
                    print "Best Result: %f" %(bestResult)
                    print "Previous Result: %f\n\n" %(result)
                    print "This is run %i/%i" %(count,totalRuns)
                    print "Time elapsed: %f/%f minutes" %(round(currentTime/60,2),round(currentTime*totalRuns/count/60,2))
                    print "Time elapsed: %f/%f hours" %(round(currentTime/3600,2),round(currentTime*totalRuns/count/3600,2))
                    print "lr: %f \nActivation Function: %s \nNodes: %i \nMBS: %i \nLength: %i" %(lr,func,depth,mbs,leng)
                    print "=======================================================\n"

