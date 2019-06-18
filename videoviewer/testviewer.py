#run with xvfb-run -s "-ac -screen 0 1280x1024x24" python testviewer.py

import minerl
import itertools
import gym
import numpy as np
from gym.envs.classic_control import rendering

def view_obs(viewer, obs):
    viewer.imshow(obs)
    return

def make_data(env, num_epochs, max_sequence_len, sequence_limit=None, required_shape=(64,64,3)):
    #makes MineRL demonstration data in specified environment and returns path to the data
    #Data is accessed by making a dataset from one of the minerl environments 
    #and iterating over it using one of the iterators provided by the minerl.data.DataPipeline

    #num_workers is number of files to load at once. Defaults to 4..
    data = minerl.data.make('MineRLObtainDiamond-v0', num_workers=4)

    #Epochs are number of times that all files in an environment are iterated over
    #So, if you specify  num_epochs= 1, it should go through every file in the dataset once


    #You can iterate over a fixed number of samples using:
    #      for obs, rew, done, act in itertools.islice(data.seq_iter(num_epochs=1,max_sequence_len=32), 6000):
    # Iterate through a single epoch gathering sequences of at most 32 steps
    total_obs=[]
    for obs, rew, done, act in itertools.islice(data.seq_iter(num_epochs=num_epochs,\
                                                                max_sequence_len=max_sequence_len), sequence_limit):
        
        total_img = []
        for image in obs['pov']:
            if image.shape == required_shape:
                total_img.append(image)
                print("image stored!")
            else:
                print("throwing out image of incorrect shape!")
        total_obs.append(total_img)
    return total_obs


viewer = rendering.SimpleImageViewer()
total_obs = make_data('MineRLObtainDiamond-v0', num_epochs=1, max_sequence_len = 1, sequence_limit=1, required_shape=(64,64,3))
for obs in total_obs:
    viewer.imshow(obs)

viewer.close()

#print(list(obs)) #equipped items, inventory, pov

#print(obs['pov'])

#print(len(obs)(pov)) #32

#print(obs['pov'][0].shape) #(64,64,3) , BUT there are some higher resolution data that snuck into dataset!


