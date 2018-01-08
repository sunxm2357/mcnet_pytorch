# mcnet_pytorch
## Paper
It is the pytorch re-implementation of ICLR2017 paper [Decomposing Motion and Content for Natural Video Sequence Prediction](https://openreview.net/pdf?id=rkEFLFqee) by Ruben Villegas, Jimei Yang, Seunghoon Hong, Xunyu Lin and Honglak Lee.

Follow the following instruction to use it.

## Install dependences
The code is in Python 2 and you need [pip](https://pip.pypa.io/en/stable/installing/) and [homebrew](https://brew.sh/) to install dependences if you use Mac Os. Otherwise, you need [pip](https://pip.pypa.io/en/stable/installing/) and make sure you have [opencv](https://opencv.org/) installed if you are using Linux.

Suggest to use [virtualenv](https://virtualenv.pypa.io/en/stable/) and install all the dependences in this virtual environment to avoid the confliction with your original environment. For Mac Os, all dependencies and opencv can be installed by the following commands.


```
brew install opencv3
pip install -r requirements.txt
```

## Train and Test
After installation of all dependencies, we can train a model by executing train.py and assigning arguments. 

For example, we train and test model on the KTH dataset as follows by the following command

```
python train.py --nepoch=603 --nepoch_decay=0 --data=KTH --gpu_id=0 --c_dim=1 --dataroot=./data/KTH --K=10 --T=10 --name=kth_10_10 --textroot=videolist --batch_size=8 --image_size 128
python test.py --name=kth_10_10 --data=KTH --gpu_ids=0 --c_dim=1 --K=10 --T=20 --dataroot=./data/KTH --textroot=videolist --image_size 128
```

The code allows us to train and test it on [KTH](http://www.nada.kth.se/cvap/actions/) and [UCF101](http://www.nada.kth.se/cvap/actions/) datasets. If you want to try your own dataset, you will need to write a dataloader class under the data directory and add it to data methods.

All the arguments can be found under the option directory and make sure you set them as you wish.

## Visualization and Results
### Visualization
During the training, losses and validation results can be visualized with tensorboard.

```
# tensorboard --logdir=tb/<experiment-name>
tensorboard --logdir=tb/ktn_10_10
```
### Results
#### Quantitative results



