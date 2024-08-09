from env.multi_hydrogen_recharge import MultiHydrogenRecharge

env = MultiHydrogenRecharge(num_vehicles=4, num_commands=4)

len(env._get_observation())