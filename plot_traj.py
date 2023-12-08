import matplotlib.pyplot as plt
import numpy as np

robots_traj = np.load('robots_traj.npy')

plt.figure(figsize=(12, 12))
plt.plot(robots_traj[:,1], -robots_traj[:,0], '.', markersize=2)
for i in range(16):
    plt.plot(robots_traj[i,1], -robots_traj[i,0], '*', markersize=20)
plt.show()