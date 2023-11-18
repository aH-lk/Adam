from stable_baselines3 import SAC
import gymnasium as gym
import sys

env = gym.make('Humanoid-v4')
SAC_VER = int(sys.argv[1].replace('SAC_', ""))
model = SAC.load(f'saved_models/SAC_{str(SAC_VER)}.zip', env=env)
TIMESTEPS = 25000
iters = 0

while True:
    iters += 1

    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False)
    model.save(f"{'saved_models'}/{'SAC'}_{(TIMESTEPS * iters) + SAC_VER}")
