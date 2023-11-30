import asyncio
from coppeliasim_zmqremoteapi_client.asyncio import RemoteAPIClient
from e_puck_controller import EPuckController
import numpy as np

class ShapedSwarmEnv():
    def __init__(self, env_config):
        self.n_bots = env_config['n_bots']
        self.event_loop = asyncio.get_event_loop() 
        self.client = RemoteAPIClient()
        # based on radius of e-puck, the control point is set as the point where
        # the perpendicular bisector of the axle intersects the circumference of robot
        # self.controller = EPuckController(35 * 1e-3)
        self.controller = EPuckController(2 * 1e-3) # hard controller
        self.obs = None

    def __enter__(self):
        self.event_loop.run_until_complete(self.client.__aenter__())
        self.event_loop.run_until_complete(self._init_simulation())
        return self
    
    def __exit__(self, *args):
        self.event_loop.run_until_complete(self._close_simulation())
        self.event_loop.run_until_complete(self.client.__aexit__(*args))
        return False
    
    def step(self, target_vels):
        if self.obs is not None:
            self.event_loop.run_until_complete(self._command_swarm(target_vels, self.obs))
        self.event_loop.run_until_complete(self.client.step())
        self.obs = self.event_loop.run_until_complete(self._get_swarm_state())
        self.obs = self.controller.extension_state(self.obs)
        return self.obs

    async def _get_swarm_state(self):
        # cheaply getting the ground truth positions and orientations of the bots from simulation
        # this localization should be processed from an overhead camera for closer to reality simulation
        tasks = [self.sim.getObjectPosition(self.bots_joints[j][0], self.sim.handle_world) for j in range(self.n_bots)]
        tasks += [self.sim.getObjectPosition(self.bots_joints[j][1], self.sim.handle_world) for j in range(self.n_bots)]
        results = await asyncio.gather(*tasks)
        left_joint_positions = np.asarray(results[:int(len(results)/2)])
        right_joint_positions = np.asarray(results[int(len(results)/2):])

        positions = (left_joint_positions + right_joint_positions) / 2              
        # retrive yaw angles in the range of -pi to pi       
        orientations = np.arctan2(-(left_joint_positions[:,0] - right_joint_positions[:,0]),
                                   (left_joint_positions[:,1] - right_joint_positions[:,1]))       
        # return x, y, yaw
        return np.hstack((positions[:,:2], orientations[:,np.newaxis]))
    
    async def _command_swarm(self, target_vels, curr_states):
        tasks = []
        for j, (target_vel, curr_state) in enumerate(zip(target_vels, curr_states)):
            wheel_omg_l, wheel_omg_r = self.controller.get_wheel_commands(target_vel, curr_state)
            tasks.append(self.sim.setJointTargetVelocity(self.bots_joints[j][0], wheel_omg_l))
            tasks.append(self.sim.setJointTargetVelocity(self.bots_joints[j][1], wheel_omg_r))
        await asyncio.gather(*tasks)
    
    async def _init_simulation(self):
        self.sim = await self.client.getObject('sim')
        self.defaultIdlsFps = await self.sim.getInt32Param(self.sim.intparam_idle_fps)
        await self.sim.setInt32Param(self.sim.intparam_idle_fps, 0)
        tasks = [self.sim.getObject(f'/ePuck[{j}]/leftJoint') for j in range(self.n_bots)]
        left_joints = await asyncio.gather(*tasks)
        tasks = [self.sim.getObject(f'/ePuck[{j}]/rightJoint') for j in range(self.n_bots)]
        right_joints = await asyncio.gather(*tasks)
        self.bots_joints = {j:(left_joints[j], right_joints[j]) for j in range(self.n_bots)}
        await self.client.setStepping(True)
        await self.sim.startSimulation()

    async def _close_simulation(self):
        await self.sim.stopSimulation()
        await self.sim.setInt32Param(self.sim.intparam_idle_fps, self.defaultIdlsFps)
