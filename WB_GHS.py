import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
from wb import *

imgs = getImages("./5D - part 1/")
folder = "./Results/"
count = 0
for img in imgs:
    count += 1
    g = calcGamma(img)
    img = adjust_gamma(img,g)
    k = constraint(img)
    imgUp = updImg(img,k)
    img2 = shiftHist(imgUp)
    saveImg(img2,os.path.join(folder, "result" + str(count)))
