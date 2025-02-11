from env.multi_hydrogen_recharge import MultiHydrogenRecharge

env = MultiHydrogenRecharge(num_vehicles=2)
for i in range(0, 10):
    state = env.reset()

    for i in range(0, 5):
        next_state, reward, done = env.step([[1, 1, 1], [2, 2, 2]])

        # print(f"Next State: {next_state}")
        # print(f"Reward: {reward}")
        # print(f"Done: {done}")

        # print(f"Number of Commands: {env.num_commands}")
        # print(env.num_commands)