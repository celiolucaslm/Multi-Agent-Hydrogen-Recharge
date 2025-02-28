from env.multi_hydrogen_recharge import MultiHydrogenRecharge

env = MultiHydrogenRecharge(num_vehicles=5, seed=42)

state = env.reset()

print(state)

env.step()