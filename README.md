# MobileComputing Project : Walk a dog

The following repository is the source code for our novel TTA model that predicts canine activities via smartwatch accelerometer, gyroscopic sensors.
With it, we can estimate with extreme accuracy, the amount of calories that was burned(by the dog) during the dog walk.


### Prerequisites

To run our code as it is, the following directories are required in the root directory

    $ mkdir dataset
    $ mkdir log

Within the dataset folder, place the dataset to be used


### Environment Settings

Install our env setting through conda

    $ conda env create -f dogwalk_env.yml


### Training and Evaluation

To train and adapt our models, run the following code

    $ . script/active/source_gen.sh
    $ . script/active/adaptation_gen.sh
    

### Evaluation

To format the results obtained from the training above, run the following code 

    $ . script/active/evaluation_gen.sh
