import training_nets as tn
import support_functions as sf

# This block of code is optional. Do not fill if you run the project
# from inside the project directory.
baseSavePath=''#'/home/eichmeierbr/Documents/CS5600/project2/'

####################################################################################
# This block of code is used to obtain the stock data.
# Format the dates as YYYY/MM/DD
# Match the ticker using all caps to its NYSE ticker.
####################################################################################
startDate = '2017-05-01'
endDate = '2017-12-31'
ticker = 'AAPL'
apple = sf.getStock(ticker, startDate, endDate)


####################################################################################
#
# This block has parameters that are used in multiple of the network functions.
#
# memory represents the number of days being used to predict the 
#       next value. The conv nets are designed to be square shaped. 
#       As such, only use perfect squares (4, 25, 64, etc)
#
# networkStyle must be either 'ann', or 'conv'
#
# numberOfLayers represents the depth of the net. Only input 1-3
#
# version is used to save various nets. The number will be added
#       to the save name.
#
####################################################################################

epochs = 1
version = 20

def parameters():

    memory = 25
    networkStyle = 'ann'
    numberOfLayers = 2
    saveName=networkStyle+'%iLayer_%iV%i' %(numberOfLayers,memory,version)

    return memory,networkStyle,numberOfLayers, saveName


def massTestingParameters():
    layers = [1,2,3]
    netStyles=['ann','conv']
    memories = [25]
    return layers, netStyles,memories

####################################################################################
#
# Unit test for the cyclical testing. It tests all combinations of network
# architectures (1, 2, and 3 layers. ConvNets and ANNs)
#
# If you want to add your own parameters to test, modify the first two blocks of 
# parameters following the instructions in the previous comment.
# 
# This testing only requires a couple of minutes with few parameters and epochs=1
#
####################################################################################


def unitTestCycleTesting():
    learningRates = [.001]
    activationFunctions = ['relu','linear']
    layerSize = [20]
    mbs = [10]

    layers, netStyles,memories=massTestingParameters()

    for memory in memories:
        for numberOfLayers in layers:
            for networkStyle in netStyles:
                saveName=networkStyle+'%iLayer_%iV%i' %(numberOfLayers,memory,version)
                tn.initiateTraining(apple,learningRates,activationFunctions,layerSize,mbs,epochs,memory,networkStyle,numberOfLayers,saveName,baseSavePath)


####################################################################################
#
# This block of unit tests contain the tests for making predictions.
#
####################################################################################


# This function makes predictions from the networks created in the unitTestCycleTesting
def unitTestPredictions():

    layers, netStyles,memories=massTestingParameters()
    
    for memory in memories:
        for numberOfLayers in layers:
            for networkStyle in netStyles:
                saveName=networkStyle+'%iLayer_%iV%i' %(numberOfLayers,memory,version)
                tn.predictTomorrow(apple,networkStyle,saveName,numberOfLayers)


# This function predicts every target value in the testing data for 
# a single network. Match the network with the parameters in the parameter
# function.
# Note: The functions given spanned 5/1/17-12/31/17. Altering these
# dates may slightly alter the reported accuracy.
def confirmAccuracy():

    memory,networkStyle,numberOfLayers, saveName=parameters()

    accuracy=tn.confirmTestingAccuracy(apple,networkStyle,saveName,numberOfLayers)
    print 'Net Accuracy: %f' %(accuracy)
    return accuracy

# This function predicts a value for a single network
def predictSingleNet(style,layer,vers,mem):
    save_name=style+'%iLayer_%iV%i' %(layer,mem,vers)
    tn.predictTomorrow(apple,style,save_name,layer)

# This function makes a prediction using all of the provided networks.
def predictCurrentNets():

    predictSingleNet('conv',1,2,49)
    predictSingleNet('conv',2,1,49)
    predictSingleNet('conv',2,2,49)
    predictSingleNet('conv',3,1,25)
    predictSingleNet('conv',3,1,36)

    predictSingleNet('ann',1,2,25)
    predictSingleNet('ann',2,3,25)
    predictSingleNet('ann',3,3,25)


####################################################################################
#
# This section contains the unit test for training individual nets.
#   To begin training, set the desired parameters for MBS, LR, activation 
#   functions, and the depths for each layer. The funcs and depths must be
#   input as a single [] container.
#   
####################################################################################

def unitTestSingleNet():
    MBS = 20
    LR = 0.0001
    funcs=['tanh','tanh']
    depths=[200,200]

    memory,networkStyle, numberOfLayers, saveName=parameters()
    vers=version+1
    save_Name=networkStyle+'%iLayer_%iV%i' %(numberOfLayers,memory,vers)
    tn.train_net(apple,memory,networkStyle,epochs,MBS,LR,funcs,depths,save_Name,baseSavePath, networkStyle=='conv')
    tn.predictTomorrow(apple,networkStyle,save_Name,numberOfLayers)


unitTestCycleTesting()
unitTestPredictions()
unitTestSingleNet()
confirmAccuracy()
predictCurrentNets()
print 'All Unit Tests Are Successful'