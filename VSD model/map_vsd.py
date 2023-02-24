import torch
import torchvision
from torch import nn
# from cam import draw_cam
import re
import cv2
import os
import math
from scipy import stats
from PIL import Image
from skimage import io
from gradcam import GradCAM, GradCamPlusPlus
from skimage.metrics import structural_similarity as compare_ssim
import torchvision.models as models
import numpy as np
import scipy.io as sio
import data_loader

for i in range(4):

    folder_path_g = '/mnt/disk10T_2/fangqiuyue/xutao_processed/3_kodak_ori_png/NIMA/gmap_%d/'%(22+(i)*5)
    files_g = os.listdir(folder_path_g)
    files_g.sort()
    folder_path_gp = '/mnt/disk10T_2/fangqiuyue/xutao_processed/3_kodak_ori_png/NIMA/gppmap_%d/'%(22+(i)*5)
    files_gp = os.listdir(folder_path_gp)
    files_gp.sort()
    folder_path_s = '/mnt/disk10T_2/fangqiuyue/xutao_processed/3_kodak_ori_png/NIMA/smap_%d/'%(22+(i)*5)
    files_s = os.listdir(folder_path_s)
    files_s.sort()



    output_dir = '/mnt/disk10T_2/fangqiuyue/xutao_processed/vis_kodak_nima_new/vis904_map_tid_hyper/qp%d_map'%(22+(i)*5)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    path_mat = '/mnt/disk10T_2/fangqiuyue/kodak/pred_nima/a_best_904.mat'
    load_data = sio.loadmat(path_mat)
    a_best = load_data['a_best']
    a1 = abs(a_best[0, :])
    a2 = abs(a_best[1, :])
    a3 = abs(a_best[2, :])
    a4 = abs(a_best[3, :])
    a5 = abs(a_best[4, :])
    a6 = abs(a_best[5, :])
    a7 = abs(a_best[6, :])
    a8 = abs(a_best[7, :])
    a9 = abs(a_best[8, :])
    a10 = abs(a_best[9, :])
    a11 = abs(a_best[10, :])
    a12 = abs(a_best[11, :])
    a13 = abs(a_best[12, :])

    for i in range(24):

        im_path_g = os.path.join(folder_path_g, files_g[i])
        im_path_gp = os.path.join(folder_path_gp, files_gp[i])
        im_path_s = os.path.join(folder_path_s, files_s[i])
        num0 = files_g[i].split('-')[0]
        num = int(num0) - 1

        img_g = io.imread(im_path_g)
        img_g = np.float32(cv2.resize(img_g, (224, 224))) / 255
        img_gp = io.imread(im_path_gp)
        img_gp = np.float32(cv2.resize(img_gp, (224, 224))) / 255
        img_s = io.imread(im_path_s)
        img_s = np.float32(cv2.resize(img_s, (224, 224))) / 255

        img_g2 = img_g ** 2
        img_g2 -= np.min(img_g2)
        img_g2 /= np.max(img_g2)
        img_g3 = img_g ** 3
        img_g3 -= np.min(img_g3)
        img_g3 /= np.max(img_g3)
        img_s2 = img_s ** 2
        img_s2 -= np.min(img_s2)
        img_s2 /= np.max(img_s2)
        img_s3 = img_s ** 3
        img_s3 -= np.min(img_s3)
        img_s3 /= np.max(img_s3)
        img_gp2 = img_gp ** 2
        img_gp2 -= np.min(img_gp2)
        img_gp2 /= np.max(img_gp2)
        img_gp3 = img_gp ** 3
        img_gp3 -= np.min(img_gp3)
        img_gp3 /= np.max(img_gp3)
        mask4 = img_g * img_gp
        mask4 -= np.min(mask4)
        mask4 /= np.max(mask4)
        mask5 = img_g * img_s
        mask5 -= np.min(mask5)
        mask5 /= np.max(mask5)
        mask6 = img_gp * img_s
        mask6 -= np.min(mask6)
        mask6 /= np.max(mask6)
        mask7 = img_gp * img_s * img_g
        mask7 -= np.min(mask7)
        mask7 /= np.max(mask7)

        # map_output = (a7[i] * img_g + a8[i] * (img_g2)  + a9[i] * (img_g3) + a1[i] * img_gp + a2[i] * (img_gp2)
        #               + a3[i] * (img_gp3) + a4[i] * img_s + a5[i] * (img_s2) + a6[i] * (img_s3)) / (a1[i] + a2[i] + a3[i] + a4[i] + a5[i] + a6[i] + a7[i] + a8[i] + a9[i])
        # map_output = (a7[i] * img_g + a8[i] * (img_g2)  + a9[i] * (img_g3) + a1[i] * img_gp + a2[i] * (img_gp2) + a3[i] * (img_g3) + a4[i] * img_s + a5[i] * (img_s2) + a6[i] * (img_g3)
        #               + a10[i] * (mask4) + a11[i] * (mask5) + a12[i] * (mask6) + a13[i] * (mask7)) / (a1[i] + a4[i] + a7[i])
        map_output = (a7[i] * img_g + a8[i] * (img_g2)  + a9[i] * (img_g3) + a1[i] * img_gp + a2[i] * (img_gp2) + a3[i] * (img_gp3) + a4[i] * img_s + a5[i] * (img_s2) + a6[i] * (img_s3)
                      + a10[i] * (mask4) + a11[i] * (mask5) + a12[i] * (mask6) + a13[i] * (mask7)) / (a1[i] + a2[i] + a3[i] + a4[i] + a5[i] + a6[i] + a7[i] + a8[i] + a9[i] + a10[i] + a11[i] + a12[i]+ a13[i])
        # map_output = (a7[i] * img_g + a8[i] * (img_g2)  + a9[i] * (img_g3) + a1[i] * img_gp + a2[i] * (img_gp2) + a3[i] * (img_g3) + a4[i] * img_s + a5[i] * (img_s2) + a6[i] * (img_g3)) / (a1[i] + a4[i] + a7[i])
        # map_output = (a7[i] * img_g + a8[i] * (img_g2) + a1[i] * img_gp + a2[i] * (img_gp2) + a4[i] * img_s + a5[i] * (img_s2)
        #               + a10[i] * (mask4) + a11[i] * (mask5) + a12[i] * (mask6)) / (a1[i] + a2[i] + a4[i] + a5[i] + a7[i] + a8[i] + a10[i] + a11[i] + a12[i])
        # map_output = (a7[i] * img_g + a8[i] * (img_g2) + a1[i] * img_gp + a2[i] * (img_gp2)
        #               + a4[i] * img_s + a5[i] * (img_s2)) / (a1[i] + a2[i] + a4[i] + a5[i] + a7[i] + a8[i])
        # map_output = (a7[i] * img_g + a1[i] * img_gp + a4[i] * img_s) / (a1[i] + a4[i] + a7[i])


        map_output = np.maximum(map_output, 0)
        map_output -= np.min(map_output)
        map_output /= np.max(map_output)
        io.imsave(os.path.join(output_dir, num0+'.png'), np.uint8(255 * map_output))

        print('%s | %s | %s | %s' % (num0, files_g[i], files_gp[i], files_s[i]))


print('End!!!\n')
