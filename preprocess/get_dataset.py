import numpy as np
import cv2
import preprocess.method as method
import os
import random
from skimage import img_as_ubyte
import glob


# Initialization parameters
pixelThreshold = 30
base_path = 'your_path'
save_path = 'your_save_path'
sliceDirList = os.listdir(base_path)
RotationList = [-20, -15, -10, -5, 0]
ScaleList = [0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3]
noiseList = [0, 0.005, 0.01, 0.03]
img_list_ct = sorted(glob.glob(os.path.join(base_path, 'CT', '*.bmp')))
img_list_mr = sorted(glob.glob(os.path.join(base_path, 'MR', '*.bmp')))
name_count = 0
sliceID = 0
patchID = 0
patchFlag = np.zeros((64, 64), np.uint8)
countFlag = 0
numberOfpoint = 0

solution = method.Method()

for path_ct, path_mr in zip(img_list_ct, img_list_mr):

    ct_img = cv2.imread(path_ct)
    mr_img = cv2.imread(path_mr)

    ct_gray = cv2.cvtColor(ct_img,cv2.COLOR_BGR2GRAY)
    mr_gray = cv2.cvtColor(mr_img,cv2.COLOR_BGR2GRAY)
    point = solution.find_keypoint(mr_gray)
    random.shuffle(point)

    patches_mr = solution.genPatches(mr_gray, np.array(point))
    keypoint_tmp = []
    for i in range(patches_mr.shape[0]):
        if np.sum(patches_mr[i]) / (64 * 64) > pixelThreshold:
            keypoint_tmp.append((keypoint_tmp[i][0], keypoint_tmp[i][1]))

    patches_ct = solution.genPatches(ct_gray,np.array(keypoint_tmp))
    keypoint = []
    for i in range(patches_ct.shape[0]):
        if np.sum(patches_ct[i]) / (64 * 64) > pixelThreshold:
            keypoint.append((keypoint[i][0], keypoint[i][1]))

    finalpoint = np.array(keypoint)
    for i in range(len(finalpoint)):
        for scale in ScaleList:
            for rotation in RotationList:

                ct_mat,ct_gray_rs = solution.rot_scale(ct_gray,rotation,scale)
                mr_mat,mr_gray_rs = solution.rot_scale(mr_gray,rotation,scale)
                if solution.boundary_judge(ct_gray_rs,finalpoint[i],ct_mat) == 1 and \
                        solution.boundary_judge(mr_gray_rs,finalpoint[i],mr_mat) == 1:

                    patch_mr = solution.gen_single_patch(mr_gray_rs, finalpoint[i], mr_mat)
                    patchFlag = np.concatenate([patchFlag, patch_mr], axis=1)
                    with open('info_train.txt', 'a') as f:
                        f.write('{} {} {} {} '.format(patchID, name_count, 0, filter(str.isdigit, path_mr)))
                        f.write('{} {} {} {}'.format(finalpoint[i, 0], finalpoint[i, 1], rotation, scale,))
                        f.write('\n')
                    patchID += 1
                    if patchFlag.shape[1] == 64 * 257:
                        patches = np.zeros((1024, 1024), np.uint8)
                        patchFlag = patchFlag[:, 64:patchFlag.shape[1]]
                        for k1 in range(16):
                            patches[(k1 * 64):(k1 + 1) * 64, :] = patchFlag[:, k1 * 1024:(k1 + 1) * 1024]
                        cv2.imwrite(os.path.join(save_path, "patch" + str(countFlag).zfill(8) + '.bmp'),
                                    patches)
                        patchFlag = np.zeros((64, 64), np.uint8)
                        countFlag += 1

                    patch_ct = solution.gen_single_patch(ct_gray_rs, finalpoint[i], ct_mat)
                    patchFlag = np.concatenate([patchFlag, patch_ct], axis=1)
                    with open('info_train.txt', 'a') as f:
                        f.write('{} {} {} {} '.format(patchID, name_count, 1, filter(str.isdigit, path_ct)))
                        f.write(
                            '{} {} {} {}'.format(finalpoint[i, 0], finalpoint[i, 1], rotation, scale,))
                        f.write('\n')
                    patchID += 1
                    if patchFlag.shape[1] == 64 * 257:
                        patches = np.zeros((1024, 1024), np.uint8)
                        patchFlag = patchFlag[:, 64:patchFlag.shape[1]]
                        for k2 in range(16):
                            patches[(k2 * 64):(k2 + 1) * 64, :] = patchFlag[:, k2 * 1024:(k2 + 1) * 1024]
                        cv2.imwrite(os.path.join(save_path, "patch" + str(countFlag).zfill(8) + '.bmp'),
                                    patches)
                        patchFlag = np.zeros((64, 64), np.uint8)
                        countFlag += 1

        name_count += 1




if patchFlag.shape[1] > 64 and patchFlag.shape[1] < 64 * 257:
    patches = np.zeros((1024, 1024), np.uint8)
    countPatches = 0
    rowsFlag = 0
    colsFlag = 0
    for i in range(patchFlag.shape[1]):
        if i != 0 and i % 64 == 0:
            patches[rowsFlag * 64:(rowsFlag + 1) * 64, colsFlag * 64:(colsFlag + 1) * 64] = patchFlag[:, i: i+64]
            colsFlag += 1
            countPatches += 1
            if countPatches % 16 == 0:
                colsFlag = 0
                rowsFlag += 1
    cv2.imwrite(os.path.join(save_path, "patch" + str(countFlag).zfill(8) + '.bmp'), patches)
