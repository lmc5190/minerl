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
        self.fc1 = nn.Linear(16 * 6 * 6 + 2, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)


    def forward(self, x, x_inventory):
        x = self.pool(F.relu(self.conv1(x))) # 64 -> 62 -> 31 
        x = self.pool(F.relu(self.conv2(x))) # 31 -> 28 -> 14
        x = self.pool(F.relu(self.conv3(x))) # 14 -> 12 -> 6
        x = x.view(-1, 16 * 6 * 6)
        x = torch.cat((x,x_inventory),1) #16*6*6 -> 16*6*6+2
        x = F.relu(self.fc1(x)) # 16*6*6+2 -> 120
        x = F.relu(self.fc2(x)) # 120 -> 84
        x = torch.sigmoid(self.fc3(x)) # 84 -> 10, can try sigmoid here?
        return x

def normalize_tensor(x,xmin,xmax,a,b):
    #transform elements of tensor x in approximate range [xmin,xmax] to [a,b]
    A = (b-a)/(xmax-xmin)
    B = a - (b-a)*xmin/(xmax-xmin)
    return A*x + B

#download if needed
#minerl.data.download('/workspace/data')

#create a data pipeline
data = minerl.data.make("MineRLNavigateDense-v0", num_workers = 4)

#set pipeline parameters
num_epochs=5
batch_size=32

#create network, put on gpu. Use multiple gpus if you see them.
net = Net()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")
  net = nn.DataParallel(net)
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
    placedirt = torch.from_numpy(act['place']).view(true_batch_size, -1).double()
    deltapitch = torch.from_numpy(act['camera'][..., 0]).view(true_batch_size, -1).double()
    deltayaw = torch.from_numpy(act['camera'][..., 1]).view(true_batch_size, -1).double()  

    #normalize scalar values in tensors
    with torch.no_grad():
        img = normalize_tensor(img,0,255.0,-1.0,1.0)
        compassAngle = normalize_tensor(compassAngle,-180.0,180.0,-1.0,1.0)
        dirt = normalize_tensor(dirt,0,64.0*36,-1.0,1.0)
        deltapitch =  normalize_tensor(deltapitch,-180.0,180.0,0,1.0)
        deltayaw =  normalize_tensor(deltayaw,-360.0,360.0,0,1.0)

    #shape observations
    img=img.view(true_batch_size,3,64,64)
    inventory=torch.cat((compassAngle,dirt),1)

    #shape action tensor (true_batch_size, 10)
    action = torch.cat((attack,back,forward,jump,right,\
                        sneak,sprint,placedirt, deltapitch,deltayaw),1) 

    # forward + backward + optimize
    optimizer.zero_grad()
    img, inventory, action = img.to(device), inventory.to(device), action.to(device)
    output = net(img,inventory)
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

#COMMAND TO SAVE MODEL
torch.save(net, 'models/net_navigatedense5_withcompass.pt')

#exec(open('navigatedense.py').read())