# los import no los quites para que no haya problemas (porseaca :V)
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
from wb import *

# Comentado solo lo importante
# aca sacas la funcion de WB.py y te lista todas las imagenes
# la cambias para que haga lo que quieras
imgs = getImages("./5D - part 1/")

# carpeta de destino donde guardar las imagenes modificadas
# porque quise mantener las originales tambien
folder = "./Results/"

# contador para que la imagente tenga de nombre "img0, img1,..."
count = 0

# for legendario
for img in imgs:
    count += 1
    # Tu algoritmo aca, cambias lo que esta hasta ####
    g = calcGamma(img)
    img = adjust_gamma(img,g)
    k = constraint(img)
    imgUp = updImg(img,k)
    img2 = shiftHist(imgUp)
    ####

    # guardar imagen (funcion de WB.py
    saveImg(img2,os.path.join(folder, "result" + str(count)))
