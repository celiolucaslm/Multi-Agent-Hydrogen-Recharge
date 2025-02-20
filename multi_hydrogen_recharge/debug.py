from env.multi_hydrogen_recharge import MultiHydrogenRecharge

env = MultiHydrogenRecharge(num_vehicles=2)

state = env.reset()

state
print(env.num_commands)
env._get_observation(1)
print(env.num_commands)
for _ in range(10):
    _, reward, _ = env.step([[0.1, 0.9, 0.1], [0.1, 0.9, 0.1], [0.1, 0.9, 0.1]])
    reward