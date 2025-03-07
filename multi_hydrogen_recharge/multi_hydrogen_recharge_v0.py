from env.multi_hydrogen_recharge import MultiHydrogenRecharge
import numpy as np

env = MultiHydrogenRecharge(num_vehicles=5, seed=42)

state = env.reset()

for i in range(10):
    print("Step: ", i)
    #actions = np.random.rand(env.num_vehicles, 4)
    actions = np.array([[0., 0., 0., 1.] for _ in range(env.num_vehicles)])
    observation, rewards, done = env.step(actions)

    print("Rewards: ", rewards)