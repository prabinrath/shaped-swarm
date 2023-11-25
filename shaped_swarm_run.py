from shaped_swarm_env import ShapedSwarmEnv
import numpy as np

def main():
    env_config = dict(
        n_bots=15,
    )
    with ShapedSwarmEnv(env_config) as env:
        SIM_TIME_SEC = 10
        SIM_STEP = 50 * 1e-3
        count = 0
        while count <= SIM_TIME_SEC/SIM_STEP:
            target_vels = np.zeros((15, 2))
            target_vels[:,0] = 0.05
            target_vels[:,1] = 0.05

            next_state = env.step(target_vels)
            count+=1

main()
