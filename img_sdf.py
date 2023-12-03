import cv2
import numpy as np
from scipy.spatial.distance import cdist

class ImageSdf():
    def __init__(self, img_path, scale = 0.05, range=(0.7, 0.7), coverage_r = 0.1):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        assert img is not None, "file could not be read, check with os.path.exists()"
        assert img.shape[0] == img.shape[1] == 400, "expecting a square image of shape 400x400"
        self.img_res = img.shape[0]
        _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
        kernel = np.ones((5,5),np.uint8)
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        img = 255 - img
        self.sdf = cv2.distanceTransform(img, maskSize=cv2.DIST_MASK_PRECISE, distanceType=cv2.DIST_L2)
        # self.sdf_img = (self.sdf/np.max(self.sdf)*255).astype(np.uint8)
        # cv2.imshow('sdf_img', self.sdf_img)
        # cv2.waitKey(0)
        
        self.grad_x = np.gradient(self.sdf, axis=0)
        self.grad_y = np.gradient(self.sdf, axis=1)
        self.scale = scale
        self.range = range
        self.coverage_r = coverage_r
        self.coverage_angles = np.linspace(0,2*np.pi,8)
    
    def get_gradient(self, robot_x, robot_y):
        x = int((2*robot_x/self.range[0])*200)
        y = int((2*robot_y/self.range[1])*200)
        x, y = max(min(x, 199), -199), max(min(y, 199), -199)
        x += 200
        y += 200
        gx = self.grad_x[x,y]
        gy = self.grad_y[x,y]
        norm = (gx**2+gy**2)**0.5
        return gx, gy, norm

    def shape_gradient(self, robot_x, robot_y):
        gx, gy, norm = self.get_gradient(robot_x, robot_y)
        if norm > 0:
            gx = gx/norm
            gy = gy/norm
            # return -gx*self.scale -self.scale*0.5*gy, -gy*self.scale +self.scale*0.5*gx
            return -gx*self.scale, -gy*self.scale
        else:
            return 0.0, 0.0

    def calculate_distances(self, curr_state):
        distances = cdist(curr_state[:,:2], curr_state[:,:2])
        np.fill_diagonal(distances, np.inf)
        return distances

    def coverage_gradient(self, robot_loc, nn_loc):
        max_grad = None
        max_norm = 0
        for th in self.coverage_angles:
            robot_x = robot_loc[0] + self.coverage_r*np.cos(th)
            robot_y = robot_loc[1] + self.coverage_r*np.sin(th)
            gx, gy, norm = self.get_gradient(robot_x, robot_y)
            if norm > max_norm:
                max_norm = norm
                max_grad = [gx, gy]
        
        dir_away = (robot_loc - nn_loc)[:2]
        if max_norm > 0:
            grad_dir_1 = np.asarray([-max_grad[1], max_grad[0]]) / max_norm
            grad_dir_2 = np.asarray([max_grad[1], -max_grad[0]]) / max_norm
            if np.dot(grad_dir_1, dir_away) > 0:
                return self.scale*grad_dir_1
            return self.scale*grad_dir_2