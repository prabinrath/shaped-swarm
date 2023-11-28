import cv2
import numpy as np

img = cv2.imread('art/ninja.png', cv2.IMREAD_GRAYSCALE)
assert img is not None, "file could not be read, check with os.path.exists()"
_, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
kernel = np.ones((5,5),np.uint8)
img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=2)
img = 255 - img
sdf = cv2.distanceTransform(img, maskSize=cv2.DIST_MASK_PRECISE, distanceType=cv2.DIST_L2)
sdf_img = (sdf/np.max(sdf)*255).astype(np.uint8)
cv2.imshow('sdf', sdf_img)
cv2.waitKey(0)