import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2

image_array = cv2.imread('results/U_sdf.png', cv2.IMREAD_GRAYSCALE)

# Normalize the pixel values to the range [0, 1]
normalized_image = image_array / 255.0

# Create x and y coordinates
x = np.linspace(0, 399, 400)
y = np.linspace(0, 399, 400)
x, y = np.meshgrid(x, y)

# Plot the surface
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(x, y, normalized_image, cmap='jet')

# Add a color bar which maps values to colors
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()
