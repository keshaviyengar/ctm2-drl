#!/usr/bin/env python

from math import sin, cos, asin, atan2, sqrt
import numpy as np


class IterativeIKinematics:
        def __init__(self, l_tip, k, goal_tolerance, delta, config, num_tubes):
            self.l_tip = l_tip
            self.k = k
            self.goal_tolerance = goal_tolerance
            self.delta = delta
            self.config = config
            self.num_tubes = num_tubes

        def compute_inverse_jacobian(self, joint_states):
            """
            Provides the numerical inverse jacobian as calculated by sympy
            :param joint: Current joint value
            :type joint:numpy.array
            :return: Matrix containing the inverse jacobian
            :rtype: numpy.ndarray
            """
            psi = 0
            phi = 0
            theta = 0
            r = 0
            gamma = 0
            distal_length = 0
            if self.config == "distal":
                gamma = joint_states[0::2]
                distal_length = joint_states[1::2]
            elif self.config == "proximal":
                psi = joint_states[0]
                phi = joint_states[1]
                theta = joint_states[2]
                r = joint_states[3]
            elif self.config == "full":
                # proximal
                psi = joint_states[0]
                phi = joint_states[1]
                theta = joint_states[2]
                r = joint_states[3]
                # distal
                gamma = joint_states[4::2]
                distal_length = joint_states[5::2]

            else:
                raise NameError

            k = self.k

            for i in range(0, self.num_tubes):
                if distal_length[i] > self.l_tip:
                    l_tip = self.l_tip
                    l = distal_length[i] - l_tip
                else:
                    l_tip = distal_length[i]
                    l = 0.0

            # computed with sympy
            jac = np.array([[(k * l_tip * sin(k * l) - cos(k * l) + 1) * cos(gamma) / k,
                             (k ** 2 * l_tip * cos(k * l) + k * sin(k * l)) * sin(gamma) / k, 0],
                            [-(-k * l_tip * sin(k * l) + cos(k * l) - 1) * sin(gamma) / k,
                             (-k ** 2 * l_tip * cos(k * l) - k * sin(k * l)) * cos(gamma) / k, 0],
                            [0, - k * l_tip * sin(k * l) + cos(k * l), 0]], np.float)


            return np.linalg.pinv(jac)

        def compute_action(self, state):
            commanded_pose = state['desired_goal']
            current_pose = state['achieved_goal']
            current_joint = state['observation'][0:-3]

            diff_pose = commanded_pose - current_pose

            # Check diff pose here, maybe use it to work out the delta value
            if self.check_difference(diff_pose):
                inv_jac = self.compute_inverse_jacobian(current_joint)
                diff_joint = self.get_diff_joint(inv_jac, diff_pose)
                new_action = diff_joint[-3:-1]
                return new_action

        def check_difference(self, diff_pose):
            """
            Checks the diff pose to see if the difference is above a threshold
            :param diff_pose: Vector containing the difference in translation and rotaiton needed to achieve the desired
            pose
            :return: True if the value is above a threshold
            """
            dist = sqrt(pow(diff_pose[0], 2) + pow(diff_pose[1], 2) + pow(diff_pose[2], 2))
            if dist > self.goal_tolerance:
                return True

            else:
                return False

        def get_diff_joint(self, inv_jac, diff_pose):
            """
            Function to get the desired joint value to converge on the desire
            d pose
            desired_joint = inv_jac * delta(commanded_pose-current_pose) + current_joint

            :param inv_jac: Inverse jacobian
            :type inv_jac: numpy.ndarray
            :param diff_pose: difference (between desired and current) pose (5 DOF) of the end effector
            :type diff_pose: numpy.ndarray
            :return: new desired joint values in rcm format
            """
            adj_diff_pose = self.delta * diff_pose.transpose()
            diff_joint = inv_jac.dot(adj_diff_pose)

            return diff_joint
