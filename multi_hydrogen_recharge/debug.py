from env.multi_hydrogen_recharge import MultiHydrogenRecharge

env = MultiHydrogenRecharge(num_vehicles=2)

state = env.reset()
print("Initial State:", state)

for i in range(10):
    print("Step:", i)
    _, reward, _ = env.step([[0.1, 0.9, 0.1], [0.1, 0.9, 0.1], [0.1, 0.9, 0.1]])
    vehicle = env.vehicles[0]
    print(vehicle.position)
    print(env._get_observation(0))