import training_nets as tn
import support_functions as sf


baseSavePath=''

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
# This block of parameters are used in multiple of the network functions.
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

epochs = 60
memory = 9
networkStyle = 'conv'
numberOfLayers = 1
version = 2

saveName=networkStyle+'%iLayer_%iV%i' %(numberOfLayers,memory,version)






####################################################################################
#
# This function contains the parameters and calls to begin
# testing all possible combinations of the parameters given.
# Please enter the parameters in [].
#
####################################################################################

def cyclicTraining():

    learningRates = [.01,.001]
    activationFunctions = ['relu','linear']
    layerSize = [5, 10, 17]
    mbs = [10,20]


    tn.initiateTraining(apple,learningRates,activationFunctions,layerSize,mbs,epochs,memory,networkStyle,numberOfLayers,saveName,baseSavePath)

####################################################################################
#
# This function tests the prediction of a single network. To run the function,
# match the parameters above to the net you want to make the prediction.
#
####################################################################################

def makePrediction():
    prediction=tn.predictTomorrow(apple,networkStyle,saveName,numberOfLayers)
    print prediction


####################################################################################
#
# This section contains the unit tests for training individual nets.
#   To begin training, set the desired parameters for MBS, LR, activation 
#   functions, and the depths for each layer. The funcs and depths must be
#   input as a single [] container.
#   
####################################################################################

def trainSingleNet():
    MBS = 10
    LR = 0.01
    funcs=['relu']
    depths=[5]
    tn.train_net(apple,memory,networkStyle,epochs,MBS,LR,funcs,depths,saveName,baseSavePath, networkStyle=='conv')
