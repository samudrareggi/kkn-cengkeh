import os
import cv2
import numpy as np
import csv

path = r"/home/reggi/Documents/Playground/Tyo/python/test"
imageArr = []
for file in os.listdir(path) :
  img = cv2.imread(os.path.abspath(path + "/" + file))
  print(os.path.abspath(path + "/" + file))
  bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
  hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
  feature = {
    "Name": file,
    "Mean HSV": np.mean(hsv),
    "Standart Deviasi": np.std(hsv)
  }
  imageArr.append(feature)
csvCol = ['Name', 'Mean HSV', 'Standart Deviasi']
csvFile = "feature.csv"
try:
  with open(csvFile, 'w') as data:
    writer = csv.DictWriter(data, fieldnames=csvCol)
    writer.writeheader()
    for datum in imageArr:
      writer.writerow(datum)
except IOError:
  print("I/O Error")