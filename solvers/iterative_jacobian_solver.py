#!/usr/bin/env python

from math import sin, cos, asin, atan2, sqrt
from solvers.jacobian_calculation import ComputeJacobianEquation
import numpy as np


class IterativeIKinematics:
        def __init__(self, l_tip, k, goal_tolerance, delta, method, num_tubes):
            self.l_tip = l_tip
            self.k = k
            self.goal_tolerance = goal_tolerance
            self.delta = delta
            self.method = method
            self.num_tubes = num_tubes
            self.jac_obj = ComputeJacobianEquation(num_tubes=num_tubes,
                                                    save_path='' + str(num_tubes) + '_tubes')

        def compute_inverse_jacobian(self, joint_states):
            """
            Provides the numerical inverse jacobian as calculated by sympy
            :param joint_states: Current joint value
            :type joint_states:numpy.array
            :return: Matrix containing the inverse jacobian
            :rtype: numpy.ndarray
            """
            if self.config == "distal":
                gamma = joint_states[0::2]
                distal_length = joint_states[1::2]
            else:
                raise NameError

            k = self.k
            l_tip = self.l_tip
            l = self.l_tip

            for i in range(0, self.num_tubes):
                if distal_length[i] > self.l_tip[i]:
                    l_tip[i] = self.l_tip[i]
                    l[i] = distal_length[i] - l_tip[i]
                else:
                    l_tip[i] = distal_length[i]
                    l[i] = 0.0

            jac = self.jac_obj.get_jacobian()
            jac = jac(k, l_tip, gamma, l)
            if self.method == 'transpose':
                ijac = np.transpose(jac)
            elif self.method == 'inverse':
                ijac = np.linalg.pinv(jac)
            else:
                print('method not set, defaulting to transpose.')
                ijac = np.transpose(jac)

            return ijac



        def compute_action(self, state):
            commanded_pose = state['desired_goal']
            current_pose = state['achieved_goal']
            current_joint = state['observation'][0:-3]

            diff_pose = commanded_pose - current_pose

            # Check diff pose here, maybe use it to work out the delta value
            if self.check_difference(diff_pose):
                inv_jac = self.compute_inverse_jacobian(current_joint)
                diff_joint = self.get_diff_joint(inv_jac, diff_pose)
                return diff_joint

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
