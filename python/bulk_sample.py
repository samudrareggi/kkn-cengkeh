import cv2
import numpy as np
from numpy.core.fromnumeric import mean
import xlsxwriter
from collections import Counter
from skimage.feature import greycomatrix, greycoprops
from skimage.measure import label, regionprops
from sklearn.cluster import KMeans
import os
import math
from scipy import stats

path = r"/home/reggi/Documents/Playground/Tyo/python/tes/"

workbook = xlsxwriter.Workbook(
    '/home/reggi/Documents/Playground/Tyo/python/features.xlsx')
worksheet = workbook.add_worksheet()

types = ['Kualitas-1', 'Kualitas-2']
amount_type = 10

# hsv_properties = ['hue', 'saturation', 'value']
glcm_properties = ['dissimilarity', 'correlation',
                   'homogeneity', 'contrast', 'ASM', 'energy']
angles = ['0', '45', '90', '135']
# shape_properties = ['eccentricity', 'metric']

worksheet.write(0, 0, 'File')
col = 1


for i in glcm_properties:
    for j in angles:
        worksheet.write(0, col, i+" "+j)
        col += 1
# for i in shape_properties:
#     worksheet.write(0, col, i)
#     col += 1
# for i in hsv_properties:
#     worksheet.write(0, col, i)
#     col += 1
worksheet.write(0, col, 'Class')
col += 1
row = 1

for i in types:
    for file in os.listdir(path+i):
        col = 0
        worksheet.write(row, col, file)
        col += 1
        print(file)
        src = cv2.imread(os.path.abspath(path + i + "/" + file), 1)
        tmp = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(tmp, 127, 255, cv2.THRESH_BINARY)
        # mask = cv2.dilate(mask.copy(), None, iterations=15)
        # mask = cv2.erode(mask.copy(), None, iterations=15)
        # b, g, r = cv2.split(src)
        # rgba = [b, g, r, mask]
        # dst = cv2.merge(rgba, 4)

        # contours, hierarchy = cv2.findContours(
        #     mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # selected = max(contours, key=cv2.contourArea)
        # x, y, w, h = cv2.boundingRect(selected)
        # png = dst[y:y+h, x:x+w]

        # gray = cv2.cvtColor(png, cv2.COLOR_BGR2GRAY)

        glcm = greycomatrix(mask,
                            distances=[5],
                            angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                            levels=256,
                            symmetric=True,
                            normed=True)
        glcm_props = [
            propery for name in glcm_properties for propery in greycoprops(glcm, name)[0]]
        for item in glcm_props:
            worksheet.write(row, col, item)
            col += 1

        # dimensions = png.shape
        # height = png.shape[0]
        # width = png.shape[1]
        # mayor = max(height, width)
        # minor = min(height, width)
        # eccentricity = math.sqrt(1-((minor*minor)/(mayor*mayor)))

        # height1 = src.shape[0]
        # width1 = src.shape[1]
        # edge = cv2.Canny(src, 100, 200)
        # k = 0
        # keliling = 0
        # while k < height1:
        #     l = 0
        #     while l < width1:
        #         if edge[k, l] == 255:
        #             keliling = keliling+1
        #         l = l+1
        #     k = k+1

        # k = 0
        # luas = 0
        # while k < height1:
        #     l = 0
        #     while l < width1:
        #         if mask[k, l] == 255:
        #             luas = luas+1
        #         l = l+1
        #     k = k+1
        # metric = (4*math.pi*luas)/(keliling*keliling)
        # shape_props = [eccentricity, metric]
        # for item in shape_props:
        #     worksheet.write(row, col, item)
        #     col += 1

        # hsv_image = cv2.cvtColor(png, cv2.COLOR_BGR2HSV)
        # height = png.shape[0]
        # width = png.shape[1]
        # H = hsv_image[:, :, 0]
        # S = hsv_image[:, :, 1]
        # V = hsv_image[:, :, 2]

        # hue = np.reshape(H, (1, height*width))
        # mode_h = stats.mode(hue[0])
        # if int(mode_h[0]) == 0:
        #     mode_hue = np.mean(H)
        # else:
        #     mode_hue = int(mode_h[0])

        # mean_s = np.mean(S)
        # mean_v = np.mean(V)
        # color_props = [mode_hue, mean_s, mean_v]
        # for item in color_props:
        #     worksheet.write(row, col, item)
        #     col += 1
        worksheet.write(row, col, i)
        row += 1
workbook.close()
