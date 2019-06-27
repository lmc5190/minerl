import gym
import minerl

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt

#class is copy and pasted from other file. SHOULD BE IMPORTED!!!!
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

#load model (called net)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
net = torch.load('models/net_navigatedense5_withcompass.pt', map_location=device)

#decision threshold
threshold=0.5

#make environment
env = gym.make('MineRLNavigateDense-v0')
obs, _ = env.reset()

done = False
net_reward = 0
reward_list = []
step_list = []

stepct=0
while not done:
    action = env.action_space.noop()

    if(stepct > 0):
        action['attack'] = 1 if output[0][0].item() >= threshold else 0
        action['back'] = 1 if output[0][1].item() >= threshold else 0
        action['forward'] = 1 if output[0][2].item() >= threshold else 0
        action['jump'] = 1 if output[0][3].item() >= threshold else 0
        action['right'] = 1 if output[0][4].item() >= threshold else 0
        action['sneak'] = 1 if output[0][5].item() >= threshold else 0
        action['sprint'] = 1 if output[0][6].item() >= threshold else 0
        action['place'] = 1 if output[0][7].item() >= threshold else 0
        action['camera'] = [normalize_tensor(output[0][8],0,1.0,-180.0,180.0).item(),\
                            normalize_tensor(output[0][9],0,1.0,-360.0,360.0).item()]

    obs, reward, done, info = env.step(action)
    env.render()

    net_reward += reward
    reward_list.append(net_reward)
    step_list.append(stepct)
    print("Total reward: ", net_reward)

    #put image on gpu. the np.flip() function deals with a strange torch bug.
    img = torch.from_numpy(np.flip(obs['pov'],axis=0).copy()).double()
    compassAngle = torch.tensor(obs['compassAngle']).view(1, -1).double()
    dirt = torch.tensor(obs['inventory']['dirt']).view(1, -1).double() 

    with torch.no_grad():
        img = normalize_tensor(img,0,255.0,-1.0,1.0).view(1,3,64,64)
        compassAngle = normalize_tensor(compassAngle,-180.0,180.0,-1.0,1.0)
        dirt = normalize_tensor(dirt,0,64.0*36,-1.0,1.0)
        inventory=torch.cat((compassAngle,dirt),1)

    img, inventory=img.to(device), inventory.to(device)



    with torch.no_grad():
        output = net(img,inventory)
    stepct=stepct+1

#plot reward
plt.plot(step_list, reward_list , label='reward')

# Show the plot
plt.show()