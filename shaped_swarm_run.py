from shaped_swarm_env import ShapedSwarmEnv
import numpy as np
from img_sdf import ImageSdf
from barrier_certificate import barrier_certificate


def main():
    imgsdf = ImageSdf('art/U.png', range=(1.5, 1.5))
    NUM_ROBOTS = 12
    env_config = dict(
        n_bots=NUM_ROBOTS,
    )
    with ShapedSwarmEnv(env_config) as env:
        SIM_TIME_SEC = 40
        SIM_STEP = 50 * 1e-3
        count = 0
        target_vels = np.zeros((NUM_ROBOTS, 2))
        while count <= SIM_TIME_SEC/SIM_STEP:
            curr_state = env.step(target_vels)
            for i in range(NUM_ROBOTS):
                gx, gy = imgsdf.get_gradient(curr_state[i][0], curr_state[i][1])
                target_vels[i,0] = gx
                target_vels[i,1] = gy
            target_vels = barrier_certificate(target_vels.T, curr_state[:,:2].T).T
            count+=1

main()
