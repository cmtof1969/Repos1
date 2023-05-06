#
sudo apt-get update

sudo apt-get upgrade

ubuntu-drivers devices

# Check appropriate driver version
# Refer to https://docs.nvidia.com/deeplearning/cudnn/support-matrix/index.html
# at 2023.05.06 the matrix is
# cudnn 8.9.1 for cuda 11.x / cuda toolkit 11.7 / nvidia driver > 450.80.02
sudo apt-get install nvidia-driver-515
sudo reboot
nvidia-smi

# install cuda tool kit

# install cudnn
## notice!
## when install libraries, one following commands are enough
## sudo apt-get install libcudnn8
## sudo apt-get install libcudnn8-dev
## sudo apt-get install libcudnn8-samples
# run a sample to test cudnn can work (run RNN)

# install anaconda
## select modify .bashrc

# create tf environment
# activate tf environment
# Update pip
pip install -U pip

# Install tensorflow
pip install tensorflow

# Install jupyter notebook
pip install notebook

