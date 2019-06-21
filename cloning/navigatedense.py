import minerl
import gym
import itertools
import numpy as np

#ABOUT DATA PIPELINE
#data.seq_iter should give you at most max_sequence_len from a single example
#- it could return less (e.g. at the end of a single demonstration)
#num_epochs is the number of epochs through through the entire dataset for a given
#experiment. So if you set num_epochs=2 you should see every human demonstration 
#2 times before the iterator runs out. Feel free to expand this interface,
#we welcome pull-requests to add additional functionality
#such as randomly sampling sequences from the entire dataset without using
#too much memory

#There are a good number of samples in the dataset - it certainly loops
#for a while #If you have more ram consider increasing the 
#number of parallel workers when making the data object, or
#move the data to a faster drive

# Sample some data from the dataset!
data = minerl.data.make("MineRLNavigateDense-v0")
num_epochs=1
max_sequence_len = 32
sequence_limit=1
required_shape = (64,64,3)
total_obs=[]
for obs, rew, done, act in itertools.islice(\
                                            data.seq_iter(num_epochs=num_epochs,\
                                            max_sequence_len=max_sequence_len),\
                                            sequence_limit):
    total_img = []
    for image in obs['pov']:
        if image.shape == required_shape:
            total_img.append(image)
            print("image stored!")
        else:
            print("throwing out image of incorrect shape!")
    total_obs.append(total_img)

#one image (64x64 pixels with 3 color channels)
#total_obs[0][0].shape
#(64, 64, 3)

# Iterate through a single epoch using sequences of at most 32 steps
#for obs, rew, done, act in data.seq_iter(num_epochs=1, batch_size=32):
    # Do something
    