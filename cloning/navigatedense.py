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

#minerl.data.download('/workspace/data')

#create a data pipeline
data = minerl.data.make("MineRLNavigateDense-v0", num_workers = 4)

#set pipeline parameters
num_epochs=1
batch_size=32

#iterate through dataset
shortseqct=0
i=0
for obs, rew, done, act in data.seq_iter(num_epochs=1, max_sequence_len=batch_size):
    i=i+1
    val=obs['compassAngle'][0][0]
    if i==1:
        minval=val
        maxval=val
    else:
        if(val < minval):
            minval=val
        if(val > maxval):
            maxval=val
    actual_sequence_len = obs['pov'].shape[0]
    if( actual_sequence_len != batch_size):
        print("need to pad sequence!")

# pad_loc is a tuple of (n_before, n_after) for each dimension,
# where (0,0) means no padding in this dimension

#!!!Pad pov
#pad_len=batch_size - actual_sequence_len
#pad_rule=((0,pad_len),(0,0),(0,0),(0,0))
#x=np.pad(obs['pov'],pad_rule, mode='edge') OR
#x=np.pad(obs['pov'],pad_rule, mode='constant')
#NEED TO TEST ZERO PADDING!!!

#!!!Pad Compass Angle

#exec(open('navigatedense.py').read())