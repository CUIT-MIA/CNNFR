

import numpy as np
import cv2
import copy
import numpy as np
from skimage import img_as_float

class Method:
    """
    this class is some method to generate training or testing data
    """
    def __init__(self):
        self.sift = cv2.xfeatures2d.SIFT_create()

    def find_keypoint(self, img):
        kp, sift_des = self.sift.detectAndCompute(img, None)
        kp_loc = []
        for i in range(len(kp)):
            kp_loc.append(kp[i].pt)
        kp_loc = list(set(kp_loc))
        return kp_loc

    def genPatches(self, img, point):
        point = point.astype(np.int)
        img_tmp = copy.deepcopy(img)

        img_pad = np.pad(img_tmp, ((32, 32), (32, 32)), 'constant', constant_values=(0, 0)).astype(np.uint8)
        patches = np.zeros((len(point), 64, 64))

        point = np.array(point) + 32

        for i in range(len(point)):
            x = point[i][0]
            y = point[i][1]
            patch_L = int(x - 32)
            patch_R = int(x + 32)
            patch_T = int(y - 32)
            patch_B = int(y + 32)
            patches[i, :, :] = img_pad[patch_L:patch_R, patch_T:patch_B]

        return patches

    def rot_scale(self, img, rot, scale):
        rows, cols = img.shape[:2]
        M = cv2.getRotationMatrix2D((cols / 2, rows / 3), rot, scale)
        warpImg = cv2.warpAffine(img, M, (cols, rows))
        return M, warpImg

    def boundary_judge(self, img, point, mat):
        if point.shape == (1, 0):
            return 0
        rows, cols = img.shape[:2]
        point = np.array(point)
        column = np.ones((point.shape[0], 1))
        point = np.column_stack((point, column))
        tmpPoint = (np.dot(mat, point.T)).T
        if tmpPoint[0, 0] >= 0 and tmpPoint[0, 0] < rows and tmpPoint[0, 1] >= 0 and tmpPoint[0, 1] < cols:
            return 1
        else:
            return 0

    def gen_single_patch(self, img, point, mat):
        rows, cols = img.shape
        img_pad = np.zeros((rows + 64, cols + 64)).astype(np.uint8)
        img_pad[32:rows + 32, 32:cols + 32] = img

        point = np.array(point)
        column = np.ones((point.shape[0], 1))
        point = np.column_stack((point, column))
        point = (np.dot(mat, point.T)).T
        point[0, 0] = point[0, 0] + 32
        point[0, 1] = point[0, 1] + 32
        Lboundary = int(point[0, 0]) - 32
        Rboundary = int(point[0, 0]) + 32
        Tboundary = int(point[0, 1]) - 32
        Bboundary = int(point[0, 1]) + 32
        patch = img_pad[Lboundary:Rboundary, Tboundary:Bboundary]
        return patch