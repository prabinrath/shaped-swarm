import numpy as np

class EPuckController():
    def __init__(self, extension):
        # source https://www.cyberbotics.com/doc/guide/epuck?version=cyberbotics:R2019a
        self.axle_length = 52 * 1e-3 # m
        self.wheel_radius = 20.5 * 1e-3 # m
        self.robot_diameter = 71 * 1e-3 # m
        self.robot_height = 50 * 1e-3 # m
        self.max_vel = 0.25 # m/s
        self.max_omg = 6.28 # rad/s
        self.robot_weight = 0.16 # kg
        assert extension > 0
        self.extension = extension
        self.ext_mat = np.array([[1.0, 0.0], [0.0, 1/extension]])
        self.rot_mat = lambda th: np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])

    def diff_drive_controller(self, vel, omg):
        # source https://wiki.ros.org/diff_drive_controller
        wheel_omg_l = (vel - omg*self.axle_length/2) / self.wheel_radius
        wheel_omg_r = (vel + omg*self.axle_length/2) / self.wheel_radius
        return wheel_omg_l[0], wheel_omg_r[0]

    def constraint_extension_controller(self, target_vel, th):
        # source lecture notes MAE 598: Multi Robot Systems (Robotarium GTech)
        cmd_vel = self.ext_mat @ self.rot_mat(-th) @ target_vel[:,np.newaxis]
        # return limited linear and angular velocities
        return min(self.max_vel, cmd_vel[0]), min(self.max_omg, cmd_vel[1])

    def get_wheel_commands(self, target_vel, curr_state):
        # target_vel: xd_cmd, yd_cmd
        # curr_state: x, y, yaw
        th = curr_state[2]
        vel, omg = self.constraint_extension_controller(target_vel, th)
        return self.diff_drive_controller(vel, omg)
    
    def extension_state(self, state):
        state[:,0] += self.extension*np.cos(state[:,2])
        state[:,1] += self.extension*np.sin(state[:,2])
        return state
