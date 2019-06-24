import minerl
import gym
import itertools

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

def normalize_tensor(x,xmin,xmax,a,b):
    #transform elements of tensor x in approximate range [xmin,xmax] to [a,b]
    #a simple linear transformation will accomplish this
    A = (b-a)/(xmax-xmin)
    B = a + (b-a)*xmin/(xmax-xmin)

    return A*x + B

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

#iterate through dataset to pull samples, shape the states and actions, and train 
shortseqct=0
i=0
for obs, rew, done, act in data.seq_iter(num_epochs=1, max_sequence_len=batch_size):
    
    true_batch_size = obs['pov'].shape[0]        

    #parse data into torch tensors
    img = torch.from_numpy(obs['pov']) # (true_batch_size, 64, 64, 3)
    compassAngle = torch.from_numpy(obs['compassAngle'][:,0]).view(true_batch_size, -1) #(true_batch_size, 1)
    dirt = torch.from_numpy(obs['inventory']['dirt']).view(true_batch_size, -1) #(true_batch_size, 1)

    #normalize tensors
    img = normalize_tensor(img,0,255,-1,1)
    compassAngle = normalize_tensor(compassAngle,-180,0,-1,1)
    dirt = normalize_tensor(dirt,0,64*36,-1,1)
    #[-1,1] for images
    #
    
    
    #print(img.shape)
    print(compassAngle.shape)
    print(dirt.shape)
    
    actual_sequence_len = obs['pov'].shape[0]
    if( actual_sequence_len != batch_size):
        print("need to pad sequence!")

# pad_loc is a tuple of (n_before, n_after) for each dimension,
# where (0,0) means no padding in this dimension



#
# Notes on State
#

#Pad pov [0,255]
#for img in obs['pov']
#    val=img
#pad_len=batch_size - actual_sequence_len
#pad_rule=((0,pad_len),(0,0),(0,0),(0,0))
#x=np.pad(obs['pov'],pad_rule, mode='edge') OR
#x=np.pad(obs['pov'],pad_rule, mode='constant')
#NEED TO TEST ZERO PADDING!!!

#Compass Angle [-180,0]
#for compassAngle in obs['compassAngle']:    
#    val=compassAngle[0]

#Inventory Dirt [0, 166] 
#could scale to max amount of dirt [0, 1] using max possible dirt 64*36
#for dirt in obs['inventory']['dirt']:
#        val=dirt

#
# Notes on Actions
#

#Set of Most Actions {0,1} binary
#Attack, back, forward, jump, right, sneak, sprint
#for actionname in act['actionname']
#   val=actionanme

#Camera Action: deltaPitch [-92.4, 71.1] and deltaYaw [-158.09666, 177]
# where  pitch is vertical angle and yaw is horizontal angle
#for camera in act['camera']:
#   pitch,yaw=camera[0],camera[1]

#Place Dirt {0,1} binary
#for place in act['place']:
    #val=place

#exec(open('navigatedense.py').read())