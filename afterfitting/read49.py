import glob

import cv2

import os

import numpy as np

from PIL import Image

import math

def test():

    number = [22,27,32,37]
    for num in number:
        save_path = './'
        file_name =  'qp'+ str(num) + '_map'
        txt_name = file_name + '.txt'
        path = save_path + file_name + '\\'
        file_name = os.listdir(path)
        file_name.sort(key=lambda x:int(x[:-4]))
        
        for name in file_name:
            print(name)
            images = Image.open(path + name)

            cols, rows = images.size 

            sample_cols = math.ceil(cols/64)

            sample_rows = math.ceil(rows/64)

            

            Value = [[0] * cols for i in range(rows)]  

            #sample_Value = [[0] * sample_cols for i in range(sample_rows)]

            sample_Value = np.zeros((64,64))

            #print(sample)
            img_array = np.array(images)
            out = np.empty((1,49))

            for x in range(0, 64):
                for y in range(0, 64):
                    sample_Value[x][y] = img_array[x, y]
            a = int(math.ceil(np.mean(sample_Value)))
            if a == 0:
                a += 1 
            out[0,0] = a 
            for x in range(64, 128):
                for y in range(0, 64):
                    sample_Value[x-64][y] = img_array[x, y]
            a = int(math.ceil(np.mean(sample_Value)))
            if a == 0:
                a += 1 
            out[0,7] = a 
            for x in range(128, 192):
                for y in range(0, 64):
                    sample_Value[x-128][y] = img_array[x, y]
            a = int(math.ceil(np.mean(sample_Value)))
            if a == 0:
                a += 1 
            out[0,14] = a 
            for x in range(192, 256):
                for y in range(0, 64):
                    sample_Value[x-192][y] = img_array[x, y]
            a = int(math.ceil(np.mean(sample_Value)))
            if a == 0:
                a += 1 
            out[0,21] = a 
            for x in range(256, 320):
                for y in range(0, 64):
                    sample_Value[x-256][y] = img_array[x, y]
            a = int(math.ceil(np.mean(sample_Value)))
            if a == 0:
                a += 1 
            out[0,28] = a 
            for x in range(320, 384):
                for y in range(0, 64):
                    sample_Value[x-320][y] = img_array[x, y]
            a = int(math.ceil(np.mean(sample_Value)))
            if a == 0:
                a += 1 
            out[0,35] = a 
            for x in range(384, 448):
                for y in range(0, 64):
                    sample_Value[x-384][y] = img_array[x, y]
            a = int(math.ceil(np.mean(sample_Value)))
            if a == 0:
                a += 1 
            out[0,42] = a 

            for x in range(0, 64):
                for y in range(64, 128):
                    sample_Value[x][y-64] = img_array[x, y]
            a = int(math.ceil(np.mean(sample_Value)))
            if a == 0:
                a += 1 
            out[0,1] = a 
            for x in range(64, 128):
                for y in range(64, 128):
                    sample_Value[x-64][y-64] = img_array[x, y]
            a = int(math.ceil(np.mean(sample_Value)))
            if a == 0:
                a += 1 
            out[0,8] = a 
            for x in range(128, 192):
                for y in range(64, 128):
                    sample_Value[x-128][y-64] = img_array[x, y]
            a = int(math.ceil(np.mean(sample_Value)))
            if a == 0:
                a += 1 
            out[0,15] = a 
            for x in range(192, 256):
                for y in range(64, 128):
                    sample_Value[x-192][y-64] = img_array[x, y]
            a = int(math.ceil(np.mean(sample_Value)))
            if a == 0:
                a += 1 
            out[0,22] = a 
            for x in range(256, 320):
                for y in range(64, 128):
                    sample_Value[x-256][y-64] = img_array[x, y]
            a = int(math.ceil(np.mean(sample_Value)))
            if a == 0:
                a += 1 
            out[0,29] = a 
            for x in range(320, 384):
                for y in range(64, 128):
                    sample_Value[x-320][y-64] = img_array[x, y]
            a = int(math.ceil(np.mean(sample_Value)))
            if a == 0:
                a += 1 
            out[0,36] = a 
            for x in range(384, 448):
                for y in range(64, 128):
                    sample_Value[x-384][y-64] = img_array[x, y]
            a = int(math.ceil(np.mean(sample_Value)))
            if a == 0:
                a += 1 
            out[0,43] = a

            for x in range(0, 64):
                for y in range(128, 192):
                    sample_Value[x][y-128] = img_array[x, y]
            a = int(math.ceil(np.mean(sample_Value)))
            if a == 0:
                a += 1 
            out[0,2] = a 
            for x in range(64, 128):
                for y in range(128, 192):
                    sample_Value[x-64][y-128] = img_array[x, y]
            a = int(math.ceil(np.mean(sample_Value)))
            if a == 0:
                a += 1 
            out[0,9] = a 
            for x in range(128, 192):
                for y in range(128, 192):
                    sample_Value[x-128][y-128] = img_array[x, y]
            a = int(math.ceil(np.mean(sample_Value)))
            if a == 0:
                a += 1 
            out[0,16] = a 
            for x in range(192, 256):
                for y in range(128, 192):
                    sample_Value[x-192][y-128] = img_array[x, y]
            a = int(math.ceil(np.mean(sample_Value)))
            if a == 0:
                a += 1 
            out[0,23] = a 
            for x in range(256, 320):
                for y in range(128, 192):
                    sample_Value[x-256][y-128] = img_array[x, y]
            a = int(math.ceil(np.mean(sample_Value)))
            if a == 0:
                a += 1 
            out[0,30] = a 
            for x in range(320, 384):
                for y in range(128, 192):
                    sample_Value[x-320][y-128] = img_array[x, y]
            a = int(math.ceil(np.mean(sample_Value)))
            if a == 0:
                a += 1 
            out[0,37] = a 
            for x in range(384, 448):
                for y in range(128, 192):
                    sample_Value[x-384][y-128] = img_array[x, y]
            a = int(math.ceil(np.mean(sample_Value)))
            if a == 0:
                a += 1 
            out[0,44] = a

            for x in range(0, 64):
                for y in range(192, 256):
                    sample_Value[x][y-192] = img_array[x, y]
            a = int(math.ceil(np.mean(sample_Value)))
            if a == 0:
                a += 1 
            out[0,3] = a 
            for x in range(64, 128):
                for y in range(192, 256):
                    sample_Value[x-64][y-192] = img_array[x, y]
            a = int(math.ceil(np.mean(sample_Value)))
            if a == 0:
                a += 1 
            out[0,10] = a 
            for x in range(128, 192):
                for y in range(192, 256):
                    sample_Value[x-128][y-192] = img_array[x, y]
            a = int(math.ceil(np.mean(sample_Value)))
            if a == 0:
                a += 1 
            out[0,17] = a 
            for x in range(192, 256):
                for y in range(192, 256):
                    sample_Value[x-192][y-192] = img_array[x, y]
            a = int(math.ceil(np.mean(sample_Value)))
            if a == 0:
                a += 1 
            out[0,24] = a 
            for x in range(256, 320):
                for y in range(192, 256):
                    sample_Value[x-256][y-192] = img_array[x, y]
            a = int(math.ceil(np.mean(sample_Value)))
            if a == 0:
                a += 1 
            out[0,31] = a 
            for x in range(320, 384):
                for y in range(192, 256):
                    sample_Value[x-320][y-192] = img_array[x, y]
            a = int(math.ceil(np.mean(sample_Value)))
            if a == 0:
                a += 1 
            out[0,38] = a 
            for x in range(384, 448):
                for y in range(192, 256):
                    sample_Value[x-384][y-192] = img_array[x, y]
            a = int(math.ceil(np.mean(sample_Value)))
            if a == 0:
                a += 1 
            out[0,45] = a

            for x in range(0, 64):
                for y in range(256, 320):
                    sample_Value[x][y-256] = img_array[x, y]
            a = int(math.ceil(np.mean(sample_Value)))
            if a == 0:
                a += 1 
            out[0,4] = a 
            for x in range(64, 128):
                for y in range(256, 320):
                    sample_Value[x-64][y-256] = img_array[x, y]
            a = int(math.ceil(np.mean(sample_Value)))
            if a == 0:
                a += 1 
            out[0,11] = a 
            for x in range(128, 192):
                for y in range(256, 320):
                    sample_Value[x-128][y-256] = img_array[x, y]
            a = int(math.ceil(np.mean(sample_Value)))
            if a == 0:
                a += 1 
            out[0,18] = a 
            for x in range(192, 256):
                for y in range(256, 320):
                    sample_Value[x-192][y-256] = img_array[x, y]
            a = int(math.ceil(np.mean(sample_Value)))
            if a == 0:
                a += 1 
            out[0,25] = a 
            for x in range(256, 320):
                for y in range(256, 320):
                    sample_Value[x-256][y-256] = img_array[x, y]
            a = int(math.ceil(np.mean(sample_Value)))
            if a == 0:
                a += 1 
            out[0,32] = a 
            for x in range(320, 384):
                for y in range(256, 320):
                    sample_Value[x-320][y-256] = img_array[x, y]
            a = int(math.ceil(np.mean(sample_Value)))
            if a == 0:
                a += 1 
            out[0,39] = a 
            for x in range(384, 448):
                for y in range(256, 320):
                    sample_Value[x-384][y-256] = img_array[x, y]
            a = int(math.ceil(np.mean(sample_Value)))
            if a == 0:
                a += 1 
            out[0,46] = a

            for x in range(0, 64):
                for y in range(320, 384):
                    sample_Value[x][y-320] = img_array[x, y]
            a = int(math.ceil(np.mean(sample_Value)))
            if a == 0:
                a += 1 
            out[0,5] = a 
            for x in range(64, 128):
                for y in range(320, 384):
                    sample_Value[x-64][y-320] = img_array[x, y]
            a = int(math.ceil(np.mean(sample_Value)))
            if a == 0:
                a += 1 
            out[0,12] = a 
            for x in range(128, 192):
                for y in range(320, 384):
                    sample_Value[x-128][y-320] = img_array[x, y]
            a = int(math.ceil(np.mean(sample_Value)))
            if a == 0:
                a += 1 
            out[0,19] = a 
            for x in range(192, 256):
                for y in range(320, 384):
                    sample_Value[x-192][y-320] = img_array[x, y]
            a = int(math.ceil(np.mean(sample_Value)))
            if a == 0:
                a += 1 
            out[0,26] = a 
            for x in range(256, 320):
                for y in range(320, 384):
                    sample_Value[x-256][y-320] = img_array[x, y]
            a = int(math.ceil(np.mean(sample_Value)))
            if a == 0:
                a += 1 
            out[0,33] = a 
            for x in range(320, 384):
                for y in range(320, 384):
                    sample_Value[x-320][y-320] = img_array[x, y]
            a = int(math.ceil(np.mean(sample_Value)))
            if a == 0:
                a += 1 
            out[0,40] = a 
            for x in range(384, 448):
                for y in range(320, 384):
                    sample_Value[x-384][y-320] = img_array[x, y]
            a = int(math.ceil(np.mean(sample_Value)))
            if a == 0:
                a += 1 
            out[0,47] = a

            for x in range(0, 64):
                for y in range(384, 448):
                    sample_Value[x][y-384] = img_array[x, y]
            a = int(math.ceil(np.mean(sample_Value)))
            if a == 0:
                a += 1 
            out[0,6] = a 
            for x in range(64, 128):
                for y in range(384, 448):
                    sample_Value[x-64][y-384] = img_array[x, y]
            a = int(math.ceil(np.mean(sample_Value)))
            if a == 0:
                a += 1 
            out[0,13] = a 
            for x in range(128, 192):
                for y in range(384, 448):
                    sample_Value[x-128][y-384] = img_array[x, y]
            a = int(math.ceil(np.mean(sample_Value)))
            if a == 0:
                a += 1 
            out[0,20] = a 
            for x in range(192, 256):
                for y in range(384, 448):
                    sample_Value[x-192][y-384] = img_array[x, y]
            a = int(math.ceil(np.mean(sample_Value)))
            if a == 0:
                a += 1 
            out[0,27] = a 
            for x in range(256, 320):
                for y in range(384, 448):
                    sample_Value[x-256][y-384] = img_array[x, y]
            a = int(math.ceil(np.mean(sample_Value)))
            if a == 0:
                a += 1 
            out[0,34] = a 
            for x in range(320, 384):
                for y in range(384, 448):
                    sample_Value[x-320][y-384] = img_array[x, y]
            a = int(math.ceil(np.mean(sample_Value)))
            if a == 0:
                a += 1 
            out[0,41] = a 
            for x in range(384, 448):
                for y in range(384, 448):
                    sample_Value[x-384][y-384] = img_array[x, y]
            a = int(math.ceil(np.mean(sample_Value)))
            if a == 0:
                a += 1 
            out[0,48] = a


            with open(save_path + txt_name,'ab') as f:
                np.savetxt(f, out, fmt = '%d', delimiter = '\t')
if __name__ == '__main__':
    test()
