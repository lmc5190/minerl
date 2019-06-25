import minerl
import gym
import itertools

import numpy as np
import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 4, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(4, 8, 4)
        self.conv3 = nn.Conv2d(8, 16, 3)
        self.fc1 = nn.Linear(16 * 6 * 6, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x))) # 64 -> 62 -> 31 
        x = self.pool(F.relu(self.conv2(x))) # 31 -> 28 -> 14
        x = self.pool(F.relu(self.conv3(x))) # 14 -> 12 -> 6
        x = x.view(-1, 16 * 6 * 6) 
        x = F.relu(self.fc1(x)) # 16*6*6 -> 120
        x = F.relu(self.fc2(x)) # 120 -> 84
        x = self.fc3(x) # 84 -> 10, can try sigmoid here?
        return x

def normalize_tensor(x,xmin,xmax,a,b):
    #transform elements of tensor x in approximate range [xmin,xmax] to [a,b]
    A = (b-a)/(xmax-xmin)
    B = a - (b-a)*xmin/(xmax-xmin)
    return A*x + B

#ABOUT DATA PIPELINE
#data.seq_iter should give you at most max_sequence_len from a single example
#- it could return less (e.g. at the end of a single demonstration)
#num_epochs is the number of epochs through through the entire dataset for a given
#experiment.

#download if needed
#minerl.data.download('/workspace/data')

#create a data pipeline
data = minerl.data.make("MineRLNavigateDense-v0", num_workers = 4)

#set pipeline parameters
num_epochs=2
batch_size=32

#create network, put on gpu
net = Net()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
net.to(device, dtype=torch.double)


#create optimizer for weights
criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

#iterate through dataset to pull samples, shape the states and actions, and train
running_loss = 0.0
i=0
for obs, rew, done, act in data.seq_iter(num_epochs=num_epochs, max_sequence_len=batch_size):
    
    true_batch_size = obs['pov'].shape[0]        

    #parse data into torch tensors
    img = torch.from_numpy(obs['pov']).double() # (true_batch_size, 64, 64, 3)
    compassAngle = torch.from_numpy(obs['compassAngle'][:,0]).view(true_batch_size, -1).double() #(true_batch_size, 1)
    dirt = torch.from_numpy(obs['inventory']['dirt']).view(true_batch_size, -1).double() 
    attack = torch.from_numpy(act['attack']).view(true_batch_size, -1).double()
    back = torch.from_numpy(act['back']).view(true_batch_size, -1).double() 
    forward = torch.from_numpy(act['forward']).view(true_batch_size, -1).double()
    jump = torch.from_numpy(act['jump']).view(true_batch_size, -1).double() 
    right = torch.from_numpy(act['right']).view(true_batch_size, -1).double() 
    sneak = torch.from_numpy(act['sneak']).view(true_batch_size, -1).double() 
    sprint = torch.from_numpy(act['sprint']).view(true_batch_size, -1).double() 
    deltapitch = torch.from_numpy(act['camera'][..., 0]).view(true_batch_size, -1).double()
    deltayaw = torch.from_numpy(act['camera'][..., 1]).view(true_batch_size, -1).double() 
    placedirt = torch.from_numpy(act['place']).view(true_batch_size, -1).double() 

    #normalize scalar values in tensors
    with torch.no_grad():
        img = normalize_tensor(img,0,255.0,-1.0,1.0)
        compassAngle = normalize_tensor(compassAngle,-180.0,0,-1.0,1.0)
        dirt = normalize_tensor(dirt,0,64.0*36,-1.0,1.0)
        deltapitch =  normalize_tensor(deltapitch,-100.0,100.0,0,1.0)
        deltayaw =  normalize_tensor(deltayaw,-180.0,180.0,0,1.0)

    #reshape image tensor
    img=img.view(true_batch_size,3,64,64)

    #shape action tensor (true_batch_size, 10)
    action = torch.cat((attack,back,forward,jump,right,\
                        sneak,sprint,placedirt, deltapitch,deltayaw),1) 

    # forward + backward + optimize
    optimizer.zero_grad()
    img, action = img.to(device), action.to(device)
    output = net(img)
    loss = criterion(output, action)
    loss.backward()
    optimizer.step()

    # print statistics
    running_loss += loss.item()
    if i % 2000 == 1999:    # print every 2000 mini-batches
        print('[%5d] loss: %.3f' %
            (i + 1, running_loss / 2000))
        running_loss = 0.0
    i=i+1

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
#attack, back, forward, jump, right, sneak, sprint
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