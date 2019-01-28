import csv
import numpy as np

from agents.ddpg_agent import DDPG
from takeoff import Takeoff

num_episodes = 1500
init_pose = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
target_pos = np.array([0., 0., 80.])

task = Takeoff(init_pose=init_pose, target_pos=target_pos)
agent = DDPG(task, prioritized_replay=False)

labels = ['episode', 'time', 'x', 'y', 'z', 'phi', 'theta', 'psi', 'x_velocity',
          'y_velocity', 'z_velocity', 'rotor_speed1', 'rotor_speed2', 'rotor_speed3', 'rotor_speed4', 'reward']
results = {x : [] for x in labels}

time = 0
save_every = 100

# Run the simulation, and save the results.
with open("DDPG_agent_stats.csv".format(), 'w') as csvfile:
    writer = csv.writer(csvfile, dialect='excel')
    writer.writerow(labels)

    run_simulation = True
    for i_episode in range(1, num_episodes+1):
        state = agent.reset_episode() # start a new episode
        while run_simulation:
            action = agent.act(state)
            next_state, reward, done = task.step(action)
            if next_state is not None:
                agent.step(action, reward, next_state, done)
                state = next_state
            else:
                done = True
                run_simulation = False
                agent.save_weights()

            if done:
                # the pose, velocity, and angular velocity of the quadcopter at the end of the episode
                print("\n===> Episode = {:4d} reward = {:}".format(i_episode, reward))
                print("pose:", task.sim.pose[:3])
                print("angles:", task.sim.pose[:3])
                print("velocity:", task.sim.v)
                print("angular velocity:", task.sim.angular_v)
                print("rotors speed:", action)


                # Write stats to file
                time = time + task.sim.time
                to_write = [i_episode] + [time] + list(task.sim.pose) + list(task.sim.v) + list(action) + [reward]
                for ii in range(len(labels)):
                    results[labels[ii]].append(to_write[ii])
                writer.writerow(to_write)

                break

