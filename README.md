 # Multi-Agent Hydrogen Recharge

 In this repository, a multi-agent reinforcement learning environment is implemented for hydrogen refueling through mobile stations (vehicles), as described in the [internship report](https://github.com/celiolucaslm/Multi-Agent-Hydrogen-Recharge/blob/main/Rapport_Stage___Célio%20MEDEIROS.pdf) (in French).
 
 The [env](https://github.com/celiolucaslm/Multi-Agent-Hydrogen-Recharge/tree/main/multi_hydrogen_recharge/env) directory contains the implementation class for the MultiHydrogenRecharge environment. The [multi_hydrogen_recharge](https://github.com/celiolucaslm/Multi-Agent-Hydrogen-Recharge/tree/main/multi_hydrogen_recharge) directory includes the classes used by the environment as well as the code for training the [MADDPG](https://arxiv.org/pdf/1706.02275) algorithm. Additionally, it contains the algorithm's test code and random action-taking code used for result comparison. 

## Example Video of the Environment in Action
 [Video](https://github.com/user-attachments/assets/e5bf792b-507e-4da9-a95d-fd3b35ca5d11)

 In this scenario, Vehicles and Commands establish their preference lists so that the matching algorithm can make assignments between them. The goal is for Vehicles to learn to prioritize the Commands they want. At each step, the Vehicle moves to the position of the serviced Command and loses hydrogen proportionally to the time of service performed.
