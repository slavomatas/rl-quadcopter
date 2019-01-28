import csv

import numpy as np

from physics_sim import PhysicsSim


class Takeoff():
    """Task (environment) that defines the goal and provides feedback to the agent."""

    def __init__(self, init_pose=None, init_velocities=None,
                 init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime)
        self.action_repeat = 3
        self.state_size = self.action_repeat * 9 # state - pose+velocity
        self.action_low = 200
        self.action_high = 700
        self.action_size = 4

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.])
        self.target_velocity = np.array([0., 0., 5.])

    def get_reward(self):

        """
        velocity_weight = np.array([-0.3, -0.3, 1.])  # reward positive z velocity while penalize abs(velocity) in x and y direction
        velocity = np.concatenate([np.abs(self.sim.v[0:2]), self.sim.v[2:3]])
        velocity_reward = np.dot(velocity_weight, velocity)
        """

        """
        position_weight = np.array([-0.3, -0.3, 1.])  # penalize displacement in x and y direction and reward positive z displacement
        position = np.concatenate([np.abs(self.sim.pose[0:2]), self.sim.pose[2:3]])
        position_reward = np.dot(position_weight, position)
        """

        #ang_velocity_weight = np.array([-0.3, -0.3, -0.3])  # penalize angles in x and y direction and reward positive z velocity
        #ang_velocity_reward = np.dot(ang_velocity_weight, np.abs(self.sim.angular_v))

        #angle_weight = np.array([-.3, -.3, -.3])  # penalize angle deviation from 0
        #angle_reward = np.dot(angle_weight, np.abs(self.sim.pose[3:]))

        velocity_reward = self.sim.v[2] #reward for vertical speed
        position_reward = self.sim.pose[2] #reward for moving in z direction

        reward = position_reward+velocity_reward

        """
        print("\n=====================================>")
        print("reward {}".format(reward))
        print("pose:", self.sim.pose[:3])
        print("angles:", self.sim.pose[:3])
        print("velocity:", self.sim.v)
        print("angular velocity:", self.sim.angular_v)
        """

        return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        state_all = []

        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds)  # update the sim pose and velocities
            reward += self.get_reward()
            #state_all.append(self.sim.pose)
            state_all.append(np.concatenate([self.sim.pose, self.sim.v]))
        next_state = np.concatenate(state_all)

        if self.sim.pose[2] > self.target_pos[2]:
            if done:
                next_state = None

        #print("rotor speed:", rotor_speeds)

        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        #state = np.concatenate([self.sim.pose] * self.action_repeat)
        state = np.concatenate([self.sim.pose, self.sim.v] * self.action_repeat)
        return state
