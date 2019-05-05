sudo apt-get update
sudo apt install curl
sudo curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
sudo python get-pip.py


export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.1.0-cp27-none-linux_x86_64.whl
sudo -H pip install $TF_BINARY_URL
sudo -H pip install tflearn


sudo -H pip install fix_yahoo_finance --upgrade --no-cache-dir
