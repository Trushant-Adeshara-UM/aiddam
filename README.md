# aiddam
EECS 504 Project Source Code

Following are the 2 papers which are followed for developing the model and getting the dataset:
1. 
2. 

## Download Dataset
For ground-truth data I used filtered csv file from the 2nd paper link to which is here.
For training the model you will need caxton_dataset_filtered.csv alongside images of those corresponding csv entries, the dataset is huge and would consume near to 50GB and one needs to manually download each data folder.


## Setup
In order to set the environment, use requirements.txt file. Current implementation is done on Ubuntu 18.04 with following versions:
- Python 3.6.9
- PyTorch 1.7.1
- Torchvision 0.8.2
- CUDA v11.3

```
virtualenv -p python3 environ
source environ/bin/activate
pip install -r requirements.txt
```

## Usage
PyTorch-Lightning (1.1.4) is used as wrapper for both the dataset and data module classes. Configuration parameters for the training are in **src/train_config.py**. 
Currently, maximum epoch are set to 15 which can be modified using (-e) and (-s) for seed parameter. Use following command to train the model:
```
python src/train.py
```

If you want to use pre-trained model, you can use following link to download the model which I trained on RTX 2080 GPU with 15 epoch in the checkpoints directory
#TODO Add Link of the model

## Results Using Sample Data
![results](./results.png)

