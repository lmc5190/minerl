#run with xvfb-run -s "-ac -screen 0 1280x1024x24" python helloworld.py

import gym
import minerl

#import logging
#for handler in logging.root.handlers:
#    logging.root.removeHandler(handler)
#logging.basicConfig(level=logging.DEBUG)

env = gym.make('MineRLNavigateDense-v0')

obs, _ = env.reset()
done = False
net_reward = 0

while not done:
        action = env.action_space.noop()
        env.render()

        action['camera'] = [0, 0.03*obs["compassAngle"]]
        action['back'] = 0
        action['forward'] = 1
        action['jump'] = 1
        action['attack'] = 1

        obs, reward, done, info = env.step(action)

        net_reward += reward
        print("Total reward: ", net_reward)
