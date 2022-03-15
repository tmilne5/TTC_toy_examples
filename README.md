# Trust-the-Critics
This repository is a PyTorch implementation of the TTC algorithm for 2-D toy examples. This code can be used to apply TTC to a variety of 2-D
source and target distributions. Unlike the main code, this will not save the trained critics; instead, figures are generated
after each critic is saved to visualize the progress of the source as it moves towards the target

## How to run this code ##
* Create a Python virtual environment with Python 3.8 installed.
* Install the necessary Python packages listed in the requirements.txt file (this can be done through pip install -r /path/to/requirements.txt).

### TTC algorithm
  

Necessary arguments for ttc_2D.py are:
* 'source' : The name of the distribution or dataset that is to be pushed towards the target (options are listed in ttc.py).
* 'target' : The name of the target dataset (options are listed in ttc.py).
* 'temp_dir' : The path of a directory where figures summarizing the training process will be saved, along with a few other files (including the log.pkl file that contains the step sizes). Despite the name, this folder isn't necessarily temporary.

Other optional arguments are described in a commented section at the top of the ttc_2D.py script. 

