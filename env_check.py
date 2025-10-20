import minerl
import gym

env = gym.make('MineRLTreechop-v0')
obs = env.reset()
print("Observation keys:", obs.keys())
print("RGB shape:", obs['pov'].shape)
env.close()
