from shaped_swarm_env import ShapedSwarmEnv
import numpy as np
from img_sdf import ImageSdf
from barrier_certificate import barrier_certificate

def main():
    NUM_ROBOTS = 16
    env_config = dict(
        n_bots=NUM_ROBOTS,
    )
    with ShapedSwarmEnv(env_config) as env:
        SIM_STEP = 50 * 1e-3
        text = 'A*S*U'
        for char in text:
            imgsdf = ImageSdf(f'art/{char}.png', range=(2.0, 2.0))
            count = 0
            target_vels = np.zeros((NUM_ROBOTS, 2))
            SIM_TIME_SEC = 20 if char=='*' else 60
            while count <= SIM_TIME_SEC/SIM_STEP:
                curr_state = env.step(target_vels)
                # nearest neighbor calculation
                nearest = np.argmin(imgsdf.calculate_distances(curr_state), axis=1)

                for i in range(NUM_ROBOTS):                   
                    gx, gy = imgsdf.shape_gradient(curr_state[i][0], curr_state[i][1])
                    gx_push, gy_push = imgsdf.coverage_gradient(curr_state[i], curr_state[nearest[i]])
                    target_vels[i,0] = gx + 0.5*gx_push
                    target_vels[i,1] = gy + 0.5*gy_push
                
                target_vels = barrier_certificate(target_vels.T, curr_state[:,:2].T).T
                count+=1

main()
