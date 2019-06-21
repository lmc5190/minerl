import minerl

#Data is accessed by making a dataset from one of the minerl environments 
#and iterating over it using one of the iterators provided by the minerl.data.DataPipeline

#download data (only need to to this once, or when data is updated!)
minerl.data.download('/workspace/data')

#num_workers is number of files to load at once. Defaults to 4..
data = minerl.data.make('MineRLObtainDiamond-v0', num_workers=4)

#Epochs are number of times that all files in an environment are iterated over
#So, if you specify  num_epochs= 1, it should go through every file in the dataset once

#You can iterate over a fixed number of samples using:
#      for obs, rew, done, act in itertools.islice(data.seq_iter(num_epochs=1,max_sequence_len=32), 6000):

# Iterate through a single epoch gathering sequences of at most 32 steps
for obs, rew, done, act in data.seq_iter(num_epochs=1, max_sequence_len=32):
    print("Number of diffrent actions:", len(act))
    #for action in act:
    #    print(act)
    print("Number of diffrent observations:", len(obs))
    #print("Number of diffrent observations:", len(obs), obs)
    #for observation in obs:
    #    print(obs)
    print("Rewards:", rew)
    print("Dones:", done)