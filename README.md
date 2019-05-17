Author: Braden Eichmeier <br />
Date: December 6, 2018 <br />
Course: CS 5600 <br />


# Introduction:

Hello, and welcome to my project. 
I hope you enjoy your time reviewing my results; 
I thoroughly enjoyed creating them.

I have provided you with all of the code used in this project,
a report detailing my findings, and some of the best performing
networks from training.

In this project, I used neural networks to predict stock prices.
The only deviations between my project and proposal include the
dates used to test the networks and implementing a portfolio
simulator. The date change is explained in the report. The 
simulator is not included due to the identified schedule risk.

To test the project, all you need to do is run the unit_test.py file.
If you want to play with the functions, I recommend you enter the 
starter.py file. Both files use simplified wrapper functions to 
run the project.

**NOTE:**  
The Yahoo Finance API has a bug in the code that causes all functionality to fail occassionaly. This bug only occurs in about 20% of the calls to the API.

If the code fails from this bug, you will see this:
1) Most recent call traceback is unit_test.py Line 16
2) The final line of the error says:
ValueError: zero-size array to reduction operation maximum which has no identity

If this error occurs, simply rerun the program. The worst this function ever inhibited my code was 3 consecutive failures. 

# Environment and Required Third-Party Libraries:

Python 2.7.15 <br />
Numpy 1.15.4 <br />
TFLEARN 1.11.0 <br />
fix_yahoo_finance 0.0.22 <br />

# Training Results Summary:

The report pdf contains all of the project findings. To summarize them, brute force testing acheives about 65% accuracy in predicting stock data. The report outlines effective hyper-parameters and displays the results of the testing.

# Included Files: 

When you downloaded the project, you received the following files.
Note: You only need to run the unit tests and the script.

### Folder: Nets
This folder contains several trained networks. These will be used during the unit tests to confirm functionality.

### Folder: Results
This folder contains the training results corresponding to the networks in the nets folder. It shows their hyperparameters and testing accuracy. Saved with the .txt files is a .pck file the code uses to load the nets.

### dependencyScript.sh
This file is a script file to install necessary dependencies. I have designed the script so the project may run on a fresh install of Ubuntu 16.04. Here is the entirety of the script:
```
sudo apt install curl
sudo curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
sudo python get-pip.py                     
                
export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.1.0-cp27-none-linux_x86_64.whl
sudo -H pip install $TF_BINARY_URL
sudo -H pip install tflearn                
                
sudo -H pip install fix_yahoo_finance --upgrade --no-cache-dir
```  

### project2Report_eichmeier.pdf
This file contains the report of the findings from training. The full record of networks is contained within this report.

### saver.py
This file contains a simple function to saave the parameters in a network. This is used to facilitate automatically loading the networks from within the training routines.

### starter.py
This file contains 3 functions:
  1) Perform cyclic training on desired hyperparameters
  2) Make a prediction from a saved network
  3) Train a single network

When running this file, please ensure the parameters in the first two sections are set correctly. Instructions are given in the file.

### support_functions.py
This file contains auxiliary functions to run the code. The functions perform data processing, file saving, and progress printing.
    
### training_nets.py
This file contains the training methods and all tflearn related functions

### unit_test.py
This file contains all the unit tests for the project. Descriptions of the tests are found in the file.
