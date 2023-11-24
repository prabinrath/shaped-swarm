from shaped_swarm_env import ShapedSwarmEnv

def main():
    env_config = {}
    with ShapedSwarmEnv(env_config) as env:
        SIM_TIME_SEC = 5
        SIM_STEP = 50 * 1e-3
        count = 0
        while count <= SIM_TIME_SEC/SIM_STEP:
            env.step()

            count+=1

main()