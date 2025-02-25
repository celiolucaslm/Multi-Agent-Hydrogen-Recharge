from env.multi_hydrogen_recharge import MultiHydrogenRecharge

env = MultiHydrogenRecharge(num_vehicles=2)

state = env.reset()
<<<<<<< HEAD
# for v in env.vehicles:
#     print("Vehicle", v.name, "Position:", v.position)

# for c in env.commands:
#     print("Command", c.name, "Position:", c.position)
=======
for v in env.vehicles:
    print("Vehicle", v.name, "Position:", v.position)

for c in env.commands:
    print("Command", c.name, "Position:", c.position)
>>>>>>> 503ba029d5bac053e3b724a0dfdd4e090516968f

for i in range(10):
    print("Step:", i)
    print(env.num_commands)
    print(env._get_observation(0))
    _, reward, _ = env.step([[0.1, 0.9, 0.1], [0.1, 0.9, 0.1], [0.1, 0.9, 0.1]])
<<<<<<< HEAD

    # for v in env.vehicles:
    #     print("Vehicle", v.name, "Position:", v.position)

    # for c in env.commands:
    #     print("Command", c.name, "Weights:", c.weights)
=======
    for v in env.vehicles:
        print("Vehicle", v.name, "Position:", v.position)

    for c in env.commands:
        print("Command", c.name, "Position:", c.position)
>>>>>>> 503ba029d5bac053e3b724a0dfdd4e090516968f
