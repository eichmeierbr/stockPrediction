import tflearn
import tensorflow as tf
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
import time
import pickle as cPickle
import numpy as np
import math
import support_functions as sf



# Function to load a persisted net
def load_net(build, path):
    network=build
    model=tflearn.DNN(network)
    model.load(path)
    return model


####################################################################################
#
# This section contains the build functions for the conv nets.
#
####################################################################################

def convnet_one_level(length,depth,lr,func,loss='mean_square',opt='adam',drop_rate=1,final='softmax'):
    size=int(math.sqrt(length)/2)+1
    input_layer = input_data(shape=[None, int(math.sqrt(length)), int(math.sqrt(length)),1])
    conv = conv_2d(input_layer, nb_filter=depth, filter_size=size, activation=func)
    pool = max_pool_2d(conv, 4)
    drop=dropout(pool,drop_rate)
    fc = fully_connected(drop, 10, activation=func)
    fc = fully_connected(fc, 2, activation=final)
    network = regression(fc, optimizer=opt,
                        loss=loss,
                        learning_rate=lr)
    return network

def convnet_two_level(length,depth,depth1, lr,func,func1, loss='mean_square',opt='adam',drop_rate=1,final='softmax'):
    size=int(math.sqrt(length)/2)+1
    input_layer = input_data(shape=[None, int(math.sqrt(length)), int(math.sqrt(length)),1])
    conv = conv_2d(input_layer, nb_filter=depth, filter_size=size, activation=func)
    pool = max_pool_2d(conv, 4)
    drop=dropout(pool,drop_rate)
    conv = conv_2d(drop, nb_filter=depth1, filter_size=size, activation=func1)
    pool = max_pool_2d(conv, 4)
    drop=dropout(pool,drop_rate)
    fc = fully_connected(drop, 10, activation=func)
    fc = fully_connected(fc, 2, activation=final)
    network = regression(fc, optimizer=opt,
                        loss=loss,
                        learning_rate=lr)
    return network

def convnet_three_level(length,depth,depth1,depth2, lr,func,func1,func2, loss='mean_square',opt='adam',drop_rate=1,final='softmax'):
        size=int(math.sqrt(length)/2)+1
        input_layer = input_data(shape=[None, int(math.sqrt(length)), int(math.sqrt(length)),1])
        conv = conv_2d(input_layer, nb_filter=depth, filter_size=size, activation=func)
        pool = max_pool_2d(conv, 4)
        drop=dropout(pool,drop_rate)
        conv = conv_2d(drop, nb_filter=depth1, filter_size=size, activation=func1)
        pool = max_pool_2d(conv, 4)
        drop=dropout(pool,drop_rate)
        conv = conv_2d(drop, nb_filter=depth2, filter_size=size, activation=func2)
        pool = max_pool_2d(conv, 4)
        drop=dropout(pool,drop_rate)
        fc = fully_connected(drop, 10, activation=func)
        fc = fully_connected(fc, 2, activation=final)
        network = regression(fc, optimizer=opt,
                            loss=loss,
                            learning_rate=lr)
        return network



####################################################################################
#
# This section contains the build functions for the ANNs.
#
####################################################################################

def ann_one_level(length,depth,lr,func,loss='mean_square',opt='adam',drop_rate=1,final='softmax'): 
    input_layer = input_data(shape=[None, length, 1])
    fc= fully_connected(input_layer, depth,
                                #weights_init=tflearn.initializations.normal(stddev=1),
                                activation=func)
    drop=dropout(fc,drop_rate)
    fc= fully_connected(drop, 2, activation=final)
    network = regression(fc, optimizer=opt,
                        loss=loss,
                        learning_rate=lr) # 0.005
    return network

def ann_two_level(length,depth,lr,func,loss='mean_square',opt='adam',drop_rate=1,final='softmax'): 
    input_layer = input_data(shape=[None, length, 1])
    fc= fully_connected(input_layer, depth[0], activation=func[0])
    drop=dropout(fc,drop_rate)
    fc= fully_connected(input_layer, depth[1], activation=func[1])
    drop=dropout(fc,drop_rate)
    fc= fully_connected(drop, 2, activation=final)
    network = regression(fc, optimizer=opt,
                        loss=loss,
                        learning_rate=lr)
    return network

def ann_three_level(length,depth,lr,func,loss='mean_square',opt='adam',drop_rate=1,final='softmax'): 
    input_layer = input_data(shape=[None, length, 1])
    fc= fully_connected(input_layer, depth[0], activation=func[0])
    drop=dropout(fc,drop_rate)
    fc= fully_connected(input_layer, depth[1], activation=func[1])
    drop=dropout(fc,drop_rate)
    fc= fully_connected(input_layer, depth[2], activation=func[2])
    drop=dropout(fc,drop_rate)
    fc= fully_connected(drop, 2, activation=final)
    network = regression(fc, optimizer=opt,
                        loss=loss,
                        learning_rate=lr)
    return network



# This function tests a net and returns the testing data accuracy
def test_model2(model, length, validX, validY,square=False):
    results = []
    for i in range(len(validX)):
        if square:
             prediction = model.predict(validX[i].reshape([-1,int(math.sqrt(length)),int(math.sqrt(length)),1]))
        else:
            prediction = model.predict(validX[i].reshape([-1, length, 1]))
        results.append(np.argmax(prediction, axis=1)[0] == \
                       np.argmax(validY[i]))
    return sum((np.array(results) == True))/float(len(results))


####################################################################################
#
# This section contains the functions that are used in brute force training.
# The functions loop through the lists of hyperparamters given for the 
# learning rates, activation functions, nodes per layer, and mini batch sizes.
# After training the function persists the most accurate net.
#
# The first function is to select the training routine based on the parameters in
# the starter file.
#
####################################################################################

def initiateTraining(stock,lr,funcs,layerDepths,mbs,epochs,memory,netStyle,numLayers,saveName,saveBase):
    if netStyle == 'ann':
        if numLayers == 1:
            cycleTesting1Layer(stock,memory,lr,funcs,layerDepths,epochs,mbs,saveName,saveBase)
        elif numLayers == 2:
            cycleTesting2Layer(stock,memory,lr,funcs,layerDepths,epochs,mbs,saveName,saveBase)
        else:
            cycleTesting3Layer(stock,memory,lr,funcs,layerDepths,epochs,mbs,saveName,saveBase)
    else:
        if numLayers == 1:
            cycleTestingConv1Layer(stock,memory,lr,funcs,layerDepths,epochs,mbs,saveName,saveBase,reshape=True)
        elif numLayers == 2:
            cycleTestingConv2Layer(stock,memory,lr,funcs,layerDepths,epochs,mbs,saveName,saveBase,reshape=True)
        else:
            cycleTestingConv3Layer(stock,memory,lr,funcs,layerDepths,epochs,mbs,saveName,saveBase,reshape=True)


def cycleTestingConv1Layer(stock,length,lrs,activations,nodes,epoch,batches,saveName,saveBase,reshape=False):

    # Obtain the stock data
    (train_d,train_t,test_d,test_t)=sf.getInputData(stock,length,reshape)

    result=0
    bestResult=0
    count=1
    totalRuns=len(lrs)*len(activations)*len(nodes)*len(batches)
    startTime=time.time()
    for lr in lrs:
        for func in activations:
            for depth in nodes:
                for mbs in batches:
                    sf.printProgress(startTime,count,totalRuns,lr,func,depth,mbs,length,result,bestResult)
                    count+=1
                    model=tflearn.DNN(convnet_one_level(length,depth,lr,func))
                    model.fit(train_d,train_t,n_epoch=epoch,shuffle=False,validation_set=(test_d,test_t),show_metric=False,batch_size=mbs)
                    result=test_model2(model,length,test_d,test_t,True)
                    if result>bestResult:
                        bestResult=result
                        bestLr=lr
                        bestFunc=func
                        bestDepth=depth
                        bestMBS=mbs
                        sf.writeConv1Level(length,epoch,bestResult,bestLr,bestFunc,bestDepth,bestMBS,saveName,saveBase)
                        model.save(saveBase+'nets/conv/oneLayer/'+saveName+'.pck')
                    tf.reset_default_graph()


def cycleTestingConv2Layer(stock,length,lrs,activations,nodes,epoch,batches,saveName,saveBase,reshape=False):

    # Obtain the stock data
    (train_d,train_t,test_d,test_t)=sf.getInputData(stock,length,reshape)

    bestResult=0
    result=0
    count=1
    totalRuns=len(lrs)*len(activations)*len(nodes)*len(batches)*len(nodes)
    startTime=time.time()
    for lr in lrs:
        for func in activations:
            for depth in nodes:
                for depth1 in nodes:
                    for mbs in batches:
                        func1=func
                        sf.printProgress2(startTime,count,totalRuns,lr,func,func1, depth,depth1, mbs,length,result,bestResult)
                        count+=1
                        model=tflearn.DNN(convnet_two_level(length,depth,depth1, lr,func,func))
                        model.fit(train_d,train_t,n_epoch=epoch,shuffle=False,validation_set=(test_d,test_t),show_metric=False,batch_size=mbs)
                        result=test_model2(model,length,test_d,test_t,True)
                        if result>bestResult:
                            bestResult=result
                            bestLr=lr
                            bestFunc=func
                            bestDepth=depth
                            bestDepth1=depth1
                            bestMBS=mbs
                            sf.writeConv2Level(length,epoch,bestResult,bestLr,bestFunc,bestFunc,bestDepth,bestDepth1, bestMBS,saveName,saveBase)
                            model.save(saveBase+'nets/conv/twoLayer/'+saveName+'.pck')
                        tf.reset_default_graph()


def cycleTestingConv3Layer(stock,length,lrs,activations,nodes,epoch,batches,saveName,saveBase,reshape=False):

    # Obtain the stock data
    (train_d,train_t,test_d,test_t)=sf.getInputData(stock,length,reshape)

    result=0
    bestResult=0
    count=1
    totalRuns=len(lrs)*len(activations)*len(nodes)*len(batches)*len(nodes)*len(nodes)
    startTime=time.time()
    for lr in lrs:
        for func in activations:
            for depth in nodes:
                for depth1 in nodes:
                    for depth2 in nodes:
                        for mbs in batches:
                            sf.printProgress3(startTime,count,totalRuns,lr,func,func,func,depth,depth1,depth2,mbs,length,result,bestResult)
                            count+=1
                            model=tflearn.DNN(convnet_three_level(length,depth,depth1,depth2, lr,func,func,func))
                            model.fit(train_d,train_t,n_epoch=epoch,shuffle=False,validation_set=(test_d,test_t),show_metric=False,batch_size=mbs)
                            result=test_model2(model,length,test_d,test_t,True)
                            if result>bestResult:
                                bestResult=result
                                bestLr=lr
                                bestFunc=func
                                bestDepth=depth
                                bestDepth1=depth1
                                bestDepth2=depth2
                                bestMBS=mbs
                                sf.writeConv3Level(length,epoch,bestResult,bestLr,bestFunc,bestFunc,bestFunc,bestDepth,bestDepth1,bestDepth2, bestMBS,saveName,saveBase)
                                model.save(saveBase+'nets/conv/threeLayer/'+saveName+'.pck')
                            tf.reset_default_graph()


def cycleTesting1Layer(stock,length,lrs,activations,nodes,epoch,batches,saveName,saveBase,reshape=False):

    # Obtain the stock data
    (train_d,train_t,test_d,test_t)=sf.getInputData(stock,length,reshape)

    result=0
    bestResult=0
    count=1
    totalRuns=len(lrs)*len(activations)*len(nodes)*len(batches)
    startTime=time.time()
    for lr in lrs:
        for func in activations:
            for depth in nodes:
                for mbs in batches:
                    sf.printProgress(startTime,count,totalRuns,lr,func,depth,mbs, length,result,bestResult)
                    count+=1
                    model=tflearn.DNN(ann_one_level(length,depth,lr,func))
                    model.fit(train_d,train_t,n_epoch=epoch,shuffle=False,validation_set=(test_d,test_t),show_metric=False,batch_size=mbs)
                    result=test_model2(model,length,test_d,test_t)

                    if result>bestResult:
                        bestResult=result
                        bestLr=lr
                        bestFunc=func
                        bestDepth=depth
                        bestMBS=mbs
                        sf.writeFile1Level(length,epoch,bestResult,bestLr,bestFunc,bestDepth,bestMBS,saveName,saveBase)
                        model.save(saveBase+'nets/ann/oneLayer/'+saveName+'.pck')
                    tf.reset_default_graph()


def cycleTesting2Layer(stock,length,lrs,activations,nodes,epoch,batches,saveName,saveBase,reshape=False):

    # Obtain the stock data
    (train_d,train_t,test_d,test_t)=sf.getInputData(stock,length,reshape)

    result=0
    bestResult=0
    count=1
    totalRuns=len(lrs)*len(activations)*len(nodes)*len(batches)*len(nodes)
    startTime=time.time()
    for lr in lrs:
        for func in activations:
            for depth in nodes:
                for depth1 in nodes:
                    for mbs in batches:
                        func1=func
                        sf.printProgress2(startTime,count,totalRuns,lr,func,func1,depth,depth1,mbs, length,result,bestResult)
                        count+=1
                        model=tflearn.DNN(ann_two_level(length,(depth, depth1),lr,(func, func1)))
                        model.fit(train_d,train_t,n_epoch=epoch,shuffle=False,validation_set=(test_d,test_t),show_metric=False,batch_size=mbs)
                        result=test_model2(model,length,test_d,test_t)
                        if result>bestResult:
                            bestResult=result
                            bestLr=lr
                            bestFunc=func
                            bestFunc1=func1
                            bestDepth=depth
                            bestDepth1=depth1
                            bestMBS=mbs
                            sf.writeFile2Level(length,epoch,bestResult,bestLr,bestFunc,bestFunc1,bestDepth,bestDepth1,bestMBS,saveName,saveBase)
                            model.save(saveBase+'nets/ann/twoLayer/'+saveName+'.pck')
                        tf.reset_default_graph()


def cycleTesting3Layer(stock,length,lrs,activations,nodes,epoch,batches,saveName,saveBase,reshape=False):

    # Obtain the stock data
    (train_d,train_t,test_d,test_t)=sf.getInputData(stock,length,reshape)

    result=0
    bestResult=0
    count=1
    totalRuns=len(lrs)*len(activations)*len(nodes)*len(batches)*len(nodes)*len(nodes)
    startTime=time.time()
    for lr in lrs:
        for func in activations:
            for depth in nodes:
                for depth1 in nodes:
                    for depth2 in nodes:
                        for mbs in batches:
                            func1=func
                            func2=func
                            sf.printProgress3(startTime,count,totalRuns,lr,func,func1,func2,depth,depth1,depth2,mbs, length,result,bestResult)
                            count+=1
                            model=tflearn.DNN(ann_three_level(length,(depth, depth1, depth2),lr,(func, func1,func2)))
                            model.fit(train_d,train_t,n_epoch=epoch,shuffle=False,validation_set=(test_d,test_t),show_metric=False,batch_size=mbs)
                            result=test_model2(model,length,test_d,test_t)
                            if result>bestResult:
                                bestResult=result
                                bestLr=lr
                                bestFunc=func
                                bestFunc1=func1
                                bestFunc2=func2
                                bestDepth=depth
                                bestDepth1=depth1
                                bestDepth2=depth2
                                bestMBS=mbs
                                sf.writeFile3Level(length,epoch,bestResult,bestLr,bestFunc,bestFunc1,bestFunc2,bestDepth,bestDepth1,bestDepth2,bestMBS,saveName,saveBase)
                                model.save(saveBase+'nets/ann/threeLayer/'+saveName+'.pck')
                            tf.reset_default_graph()




####################################################################################
#
# This section contains two functions to train a single neural network.
# The function determines the network build and save files based on the 
# parameters from the starter file.
#
####################################################################################


def getTrainingBuild(netStyle,memory,lr,depths,funcs):
    numLayers=len(depths)
    if netStyle == 'ann':
        if numLayers == 1:
            return ann_one_level(memory,depths[0],lr,funcs[0])
        elif numLayers == 2:
            return ann_two_level(memory,depths,lr,funcs)
        else:
            return ann_three_level(memory,depths,lr,funcs)
    else:
        if numLayers == 1:
            return convnet_one_level(memory,depths[0],lr,funcs[0])
        elif numLayers == 2:
            return convnet_two_level(memory,depths[0],depths[1],lr,funcs[0],funcs[1])
        else:
            return convnet_three_level(memory,depths[0],depths[1],depths[2],lr,funcs[0],funcs[1],funcs[2])

def train_net(stock,length, netStyle, epoch, mbs, lr,funcs,depths,saveName, saveBase,reshape=False): 
    
    # Make the network build
    build=getTrainingBuild(netStyle,length,lr,depths,funcs)

    # Obtain the stock data
    (train_d,train_t,test_d,test_t)=sf.getInputData(stock,length,reshape)

    # Build and train the network
    model=tflearn.DNN(build)
    model.fit(train_d,train_t,
                n_epoch=epoch,shuffle=False,
                validation_set=(test_d,test_t),
                show_metric=True,batch_size=mbs)
    tf.reset_default_graph()

    # Save the network and other training files.
    model.save('nets/'+netStyle+'/'+sf.intToString(len(funcs))+'Layer/'+saveName+'.pck')
    sf.chooseFileWriter(length,netStyle,epoch,test_model2(model,length,test_d,test_t,reshape),lr,funcs,depths,mbs,saveName,saveBase)


####################################################################################
#
# This section contains functions to predict a stock change using a trained net.
# The function prints to Buy or Sell, along with returning the predicted data.
#
####################################################################################


def predictTomorrow(stock,netStyle,netName,layers):

    # Load network parameters.
    (netStyle,memory,lr,funcs,depths) = sf.load("results/"+netStyle+"/"+sf.intToString(layers)+"Level/"+netName+".pck")

    # Obtain the prediction data.
    predictionData=sf.setPredictionData(stock[2],memory,netStyle)

    # Build the network and make a prediction.
    network=getPredictionNet(netName,netStyle,layers,memory,lr,funcs,depths)
    prediction=np.round(network.predict(predictionData))[0]
    tf.reset_default_graph()

    # Return the result
    if prediction[0] == 1.0:
        print 'Buy'
    else :
        print 'Sell'
    return prediction

def confirmTestingAccuracy(stock,netStyle,netName,layers):

    # Load network parameters.
    (netStyle,memory,lr,funcs,depths) = sf.load("results/"+netStyle+"/"+sf.intToString(layers)+"Level/"+netName+".pck")

    # Obtain the data.
    data=sf.getInputData(stock,memory,netStyle=='conv')

    # Build the network and make a prediction.
    network=getPredictionNet(netName,netStyle,layers,memory,lr,funcs,depths)
    accuracy=test_model2(network,memory,data[2],data[3],netStyle=='conv')
    tf.reset_default_graph()
    return accuracy


def getPredictionNet(netName,netStyle,numLayers,memory,lr,funcs,depths):
    if netStyle == 'ann':
        if numLayers == 1:
            loadName='nets/'+netStyle+'/oneLayer/'+netName+'.pck'
            return load_net(ann_one_level(memory,depths[0],lr,funcs[0]),loadName)
        elif numLayers == 2:
            loadName='nets/'+netStyle+'/twoLayer/'+netName+'.pck'
            return load_net(ann_two_level(memory,depths,lr,funcs),loadName)
        else:
            loadName='nets/'+netStyle+'/threeLayer/'+netName+'.pck'
            return load_net(ann_three_level(memory,depths,lr,funcs),loadName)
    else:
        if numLayers == 1:
            loadName='nets/'+netStyle+'/oneLayer/'+netName+'.pck'
            return load_net(convnet_one_level(memory,depths[0],lr,funcs[0]),loadName)
        elif numLayers == 2:
            loadName='nets/'+netStyle+'/twoLayer/'+netName+'.pck'
            return load_net(convnet_two_level(memory,depths[0],depths[1],lr,funcs[0],funcs[1]),loadName)
        else:
            loadName='nets/'+netStyle+'/threeLayer/'+netName+'.pck'
            return load_net(convnet_three_level(memory,depths[0],depths[1],depths[2],lr,funcs[0],funcs[1],funcs[2]),loadName)