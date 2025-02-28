from env.multi_hydrogen_recharge import MultiHydrogenRecharge
import numpy as np

env = MultiHydrogenRecharge(num_vehicles=5)

state = env.reset()
# for v in env.vehicles:
#     print("Vehicle", v.name, "Position:", v.position)

# for c in env.commands:
#     print("Command", c.name, "Position:", c.position)

for i in range(10):
    print("Step:", i)
    # for v in env.vehicles:
    #     if v.remaining_working_time == 0:
    #         print("Vehicle:", v.name, "Done:", done) 
    actions = np.array([[0., 1., 0.] for _ in range(env.num_vehicles)])
    state, reward, done = env.step(actions)
    
    # for c in env.commands:
    #     print("Command", c.name, "Weights:", c.weights)