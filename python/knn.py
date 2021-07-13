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
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
from sklearn.neighbors import KNeighborsClassifier
from matplotlib import pyplot as plt

init_data = '/home/reggi/Documents/Playground/Tyo/python/features.xlsx'
file_name = '/home/reggi/Documents/Playground/Tyo/python/tes/tes4.png'
dataset = pd.read_excel(init_data, engine='openpyxl')

X_train = np.array(dataset[['dissimilarity 0', 'dissimilarity 45', 'dissimilarity 90', 'dissimilarity 135', 'correlation 0', 'correlation 45', 'correlation 90', 'correlation 135', 'homogeneity 0', 'homogeneity 45',
                   'homogeneity 90', 'homogeneity 135', 'contrast 0', 'contrast 45', 'contrast 90', 'contrast 135', 'ASM 0', 'ASM 45', 'ASM 90', 'ASM 135', 'energy 0', 'energy 45', 'energy 90', 'energy 135']])
y_train = np.array(dataset[['Class']])

lb = LabelBinarizer()
y_train = lb.fit_transform(y_train).flatten()

K = 3
model = KNeighborsClassifier(n_neighbors=K).fit(X_train, y_train)

test = [[]]
ypes = ['Kualitas-1', 'Kualitas-2']
hsv_properties = ['hue', 'saturation', 'value']
glcm_properties = ['dissimilarity', 'correlation',
                   'homogeneity', 'contrast', 'ASM', 'energy']
angles = ['0', '45', '90', '135']
shape_properties = ['eccentricity', 'metric']

print(file_name)
src = cv2.imread(file_name, 1)

tmp = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
# cv2.imshow('grey', tmp)

_, mask = cv2.threshold(tmp, 127, 255, cv2.THRESH_BINARY)
# mask = cv2.dilate(mask.copy(), None, iterations=25)
# mask = cv2.erode(mask.copy(), None, iterations=1)
# cv2.imshow('thres', mask)
# b, g, r = cv2.split(src)
# rgba = [b, g, r, mask]
# dst = cv2.merge(rgba, 4)
# # cv2.imshow('dst', dst)

# contours, hierarchy = cv2.findContours(
#     mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# selected = max(contours, key=cv2.contourArea)
# x, y, w, h = cv2.boundingRect(selected)
# png = dst[y:y+h, x:x+w]
# cv2.imshow('png', png)

# cv2.waitKey(0)
# gray = cv2.cvtColor(png, cv2.COLOR_BGR2GRAY)
countLow = np.count_nonzero(mask == 0)
countHigh = np.count_nonzero(mask > 0)
print(countLow)
print(countHigh)
glcm = greycomatrix(mask,
                    distances=[5],
                    angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                    levels=256,
                    symmetric=True,
                    normed=True)
glcm_props = [
    propery for name in glcm_properties for propery in greycoprops(glcm, name)[0]]
for item in glcm_props:
    test[0].append(item)

# dimensions = mask.shape
# height = mask.shape[0]
# width = mask.shape[1]
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
#     test[0].append(item)

# hsv_image = cv2.cvtColor(mask, cv2.COLOR_BGR2HSV)
# height = mask.shape[0]
# width = mask.shape[1]
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
#     test[0].append(item)

result = model.predict(test)

# image_rgb = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
# rectangle = (300, 30, 421, 378)
# # 600, 550, 1150, 2000
# mask = np.zeros(image_rgb.shape[:2], np.uint8)

# bgdModel = np.zeros((1, 65), np.float64)
# fgdModel = np.zeros((1, 65), np.float64)

# cv2.grabCut(image_rgb, mask, rectangle, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

# mask_2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

# image_rgd_nobg = image_rgb * mask_2[:, :, np.newaxis]

# plt.imshow(image_rgd_nobg), plt.axis('off')
# plt.show()
print(test)
print(lb.inverse_transform(result))
