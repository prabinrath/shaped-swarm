import cv2
import numpy as np

class ImageSdf():
    def __init__(self, img_path, scale = 0.05, range=(0.7, 0.7)):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        assert img is not None, "file could not be read, check with os.path.exists()"
        _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
        kernel = np.ones((5,5),np.uint8)
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        img = 255 - img
        self.sdf = cv2.distanceTransform(img, maskSize=cv2.DIST_MASK_PRECISE, distanceType=cv2.DIST_L2)
        self.sdf_img = (self.sdf/np.max(self.sdf)*255).astype(np.uint8)
        # cv2.imshow('sdf', self.sdf_img)
        # cv2.waitKey(0)
        self.grad_x = np.gradient(self.sdf, axis=0)
        self.grad_y = np.gradient(self.sdf, axis=1)
        self.scale = scale
        self.range = range
    
    def get_gradient(self, x, y):
        x = int((2*x/self.range[0])*200)
        y = int((2*y/self.range[1])*200)
        x, y = max(min(x, 199), -199), max(min(y, 199), -199)
        x += 200
        y += 200
        if self.sdf[x,y] < 5.0:
            return 0.0, 0.0
        gx = self.grad_x[x,y]
        gy = self.grad_y[x,y]
        norm = (gx**2+gy**2)**0.5
        gx = gx/norm
        gy = gy/norm
        return -gx*self.scale, -gy*self.scale

