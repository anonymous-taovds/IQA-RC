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
import data_loader
from model.model import *
from cam.scorecam import *

os.environ['CUDA_VISIBLE_DEVICES'] = '3'


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        img = img.resize((224, 224))
        return img.convert('RGB')

def get_last_conv_name(net):
    """
    :param net:
    :return:
    """
    layer_name = None
    for name, m in net.named_modules():
        if isinstance(m, nn.Conv2d):
            layer_name = name
    return layer_name

def gen_cam(image, mask):
    """
    生成CAM图
    :param image: [H,W,C]
    :param mask: [H,W]
    :return: tuple(cam,heatmap)
    """
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    heatmap = heatmap[..., ::-1]  # gbr to rgb

    cam = heatmap + np.float32(image)
    # return norm_image(cam), (heatmap * 255).astype(np.uint8)
    return norm_image(cam)

def norm_image(image):
    """
    :param image: [H,W,C]
    :return:
    """
    image = image.copy()
    image -= np.max(np.min(image), 0)
    image /= np.max(image)
    image *= 255.
    return np.uint8(image)

def save_image(image_dicts, input_image_name, network, output_dir):
    prefix = os.path.splitext(input_image_name)[0]
    for key, image in image_dicts.items():
        io.imsave(os.path.join(output_dir, '{}-{}-{}.jpg'.format(prefix, network, key)), image)

def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse < 1.0e-10:
        return 100
    psnr = 20 * math.log10(1/math.sqrt(mse))
    return psnr

base_model = models.vgg16(pretrained=True)
model = NIMA(base_model).cuda()
model_path = '/model/epoch-tid.pth'
# model_path = '/model/epoch-kadid.pth'

model.load_state_dict(torch.load(model_path))
model.eval()

for i in range(4):

    # folder_path = '/mnt/disk10T_2/fangqiuyue/xutao_processed/3_tecnick_ori_png/tecnick_%d'%(19+(i)*3)
    # folder_path = '/mnt/disk10T_2/fangqiuyue/tid2013/tid_png/tid_%d'%(19+(i)*3)
    folder_path = '/mnt/disk10T_2/fangqiuyue/ori/kodak_ori_png/kodak_%d'%(22+(i)*5)
    files = os.listdir(folder_path)
    save_path = '/mnt/disk10T_2/fangqiuyue/kodak/pred_nima/'
    if not os.path.isdir(save_path):
        os.makedirs(save_path)


    id_test = np.arange(24)
    test_id = id_test.tolist()

    test_loader = data_loader.DataLoader('gfolder', folder_path, test_id, 224, 1, istrain=False)
    test_data = test_loader.get_data()



    # layer_name = 'layer4'
    layer_name = get_last_conv_name(model)
    grad_cam_plus_plus = GradCamPlusPlus(model, layer_name)
    # output_dir = '/mnt/disk10T_2/fangqiuyue/tid2013/map_all/smap_%d'%(19+(i)*3)
    output_dir = '/mnt/disk10T_2/fangqiuyue/xutao_processed/3_kodak_ori_png/smap_%d'%(22+(i)*5)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    # data_path = '/mnt/disk10T/fqy/dataset/kadid10k/ref/'
    data_path = '/mnt/disk10T/fqy/Neural-IMage-Assessment-master/20221122kodak/Kodak24/'
    # data_path = '/mnt/disk10T_2/fangqiuyue/tecnick/Tecnick'
    reference = os.listdir(data_path)
    reference.sort()

    psnr0 = []
    ssim0 = []
    psnr1 = []
    psnr00 = []
    psnr01 = []
    mos1 = []
    gt_scores = []

    item = 0
    for img0, img_name in test_data:
        item += 1
        with torch.no_grad():
            im_path = os.path.join(folder_path, img_name[0])
            name = img_name[0].split('.')
            num = int(name[0]) - 1
            ref_im_path = os.path.join(data_path, reference[num])
            img1 = io.imread(im_path)
            img1 = np.float32(cv2.resize(img1, (224, 224))) / 255
            # img = transforms(img0)
            img = torch.tensor(img0.cuda())

            output = model(img)
            output = output.view(10, 1)
            predicted_mean, predicted_std = 0.0, 0.0
            for i, elem in enumerate(output, 1):
                predicted_mean += i * elem
            pred = float(predicted_mean.item())

        #visualization
        image_dict = {}
        # Grad-CAM
        # grad_cam = GradCAM(model, layer_name)
        # mask0 = grad_cam(img)  # cam mask
        # # # image_dict['cam'] = gen_cam(img1, mask)
        # image_dict['cam-mask'] = np.uint8(255 * mask0)
        # grad_cam.remove_handlers()
        # # # save_image(image_dict, os.path.basename(im_path), 'NIMA', output_dir)

        # # # Grad-CAM++
        # grad_cam_plus_plus = GradCamPlusPlus(model, layer_name)
        # mask01 = grad_cam_plus_plus(img)  # cam mask
        # # image_dict['cam++'] = gen_cam(img1, mask)
        # image_dict['cam++-mask'] = np.uint8(255 * mask01)
        # grad_cam_plus_plus.remove_handlers()
        # save_image(image_dict, os.path.basename(im_path), 'NIMA', output_dir)

        # score-CAM
        model_dict = dict(type='none', arch=model, layer_name=layer_name, input_size=(224, 224))
        scorecam = ScoreCAM(model_dict)

        scorecam_map = scorecam(img).cpu()
        map = scorecam_map.numpy()

        mask = np.squeeze(scorecam_map)
        mask = mask.numpy()
        # mask = cam_extractor(index, output)  # cam mask
        # image_dict['scam'] = gen_cam(img1, mask)
        image_dict['scam-mask'] = np.uint8(255 * mask)
        # grad_cam_plus_plus.remove_handlers()
        save_image(image_dict, os.path.basename(im_path), 'NIMA', output_dir)
        #
        # mask = draw_cam(resnet, img0)  # cam mask
        # image_dict['cam'] = gen_cam(img1, mask)
        # save_image(image_dict, os.path.basename(im_path), 'CAM', output_dir)

        ref_img = io.imread(ref_im_path)
        ref_img = np.float32(cv2.resize(ref_img, (224, 224))) / 255
        # psnr_0 = psnr(img1, ref_img)
        # ssim_0 = compare_ssim(img1, ref_img, multichannel=True)

        # mask = np.where(mask > 0.4, 3*mask, mask)
        mask2 = mask ** 2
        mask2 -= np.min(mask2)
        mask2 /= np.max(mask2)
        mask3 = mask ** 3
        mask3 -= np.min(mask3)
        mask3 /= np.max(mask3)
        # mask4 = mask * mask0
        # mask4 -= np.min(mask4)
        # mask4 /= np.max(mask4)
        # mask5 = mask * mask01
        # mask5 -= np.min(mask5)
        # mask5 /= np.max(mask5)
        # mask6 = mask * mask01 * mask0
        # mask6-= np.min(mask6)
        # mask6 /= np.max(mask6)
        mask1 = []
        for i in range(224):
            for j in range(224):
                for k in range(3):
                    mask1.append((mask[i][j]) * ((img1[i][j][k] - ref_img[i][j][k]) ** 2))

        mse_1 = np.mean(mask1)
        # mse_1 = mse_1 * 224 * 224 / sum(map(sum, mask))
        if mse_1 < 1.0e-10:
            psnr_1 = 100
        else:
            psnr_1 = 20 * math.log10(1 / math.sqrt(mse_1))

        # if (math.isnan(psnr_1)) :
        #     psnr00.append(psnr_0)
        #     psnr01.append(psnr_1)
        # else:
        #     psnr0.append(psnr_0)
        #     psnr1.append(psnr_1)
        #     ssim0.append(ssim_0)
        #     mos1.append(mos_all[item])
        #     gt_scores = gt_scores + labels.cpu().tolist()
        print('%d:' % (item) + str(img_name[0]) + '|' + str(reference[num]) + '|%.4f|%.8f|%.4f' % (pred, mse_1, psnr_1))
        with open(os.path.join(save_path, 'wmse_scam.txt'), 'a') as f:
            f.write('%d' % (item)+ ' %.4f %.8f %.4f\n' % (pred, mse_1, psnr_1))



print('test end!!!\n')

