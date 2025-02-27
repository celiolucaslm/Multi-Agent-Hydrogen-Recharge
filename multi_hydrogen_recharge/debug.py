from env.multi_hydrogen_recharge import MultiHydrogenRecharge

env = MultiHydrogenRecharge(num_vehicles=2)

state = env.reset()
# for v in env.vehicles:
#     print("Vehicle", v.name, "Position:", v.position)

# for c in env.commands:
#     print("Command", c.name, "Position:", c.position)

for i in range(10):
    print("Step:", i)
    for v in env.vehicles:
        if v.remaining_working_time == 0:
            print("Vehicle:", v.name, "Done:", done) 
    state, reward, done = env.step([[0.1, 0.9, 0.1], [0.1, 0.9, 0.1], [0.1, 0.9, 0.1]])
    # for c in env.commands:
    #     print("Command", c.name, "Weights:", c.weights)