#run with xvfb-run -s "-ac -screen 0 1280x1024x24" python testviewer.py

#ABOUT RENDERING
#So all environments that serve pixel observations (vizdoom, startcraft, atari, minecraft)
#need to render. The process of rendering is forming that pixel level representation
#of the environment. These environments use graphics libraries like OpenGL, etc,
#to actually do the rendering at every step. When a graphics library like OpenGL tries
#to render a scene it defines a "surface" on which to render these objects.
#Some graphics libraries support surfaces hosted (and accelerated by) an X-server,
#some by Window's window manager, and even virtual surfaces such as EGL. 

#When you use a gym environment which serves you pixel observations without 
#having any window appear (unless you call env.render()), this is because the environment
#at its core uses a virtual surface. In the case of Minecraft however,
#it was built as a video game to be used with standard window manager surfaces 
#such as GLX or WGL, and therefore doesn't supprot this virtual surface rendering.
#This is out of our hands (for the moment) and so Minecraft needs a window manager to 
#render to so that it can produce pixel observations for agents. 
#When Minecraft Forge updates to 1.14 and we are able to update our fork of Malmo to 
#that version then we will be able to rewrite minecraft to be headless out of the box!
#This should result in massive speed ups and the dropping of the head (X-server)
#or virtual head (vnc, or xvfb) requirement.


#ABOUT DATA PIPELINE
#data.seq_iter should give you at most max_sequence_len from a single example
#- it could return less (e.g. at the end of a single demonstration)
#num_epochs is the number of epochs through through the entire dataset for a given
#experiment. So if you set num_epochs=2 you should see every human demonstration 
#2 times before the iterator runs out. Feel free to expand this interface,
#we welcome pull-requests to add additional functionality
#such as randomly sampling sequences from the entire dataset without using
#too much memory

#There are a good number of samples in the dataset - it certainly loops for a while
#If you have more ram consider increasing the number of parallel workers when making
#the data object, or move the data to a faster drive

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

#print(obs['pov'][0].shape) #(64,64,3) , BUT there are some higher resolution data that snuck into dataset