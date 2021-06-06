import cv2
import numpy as np
import os
from matplotlib import pyplot as plt

# Comentado solo lo importante
# Este de aca obtiene todas las images, lo hice por tamaño porque tenia otros
# archivos en la misma carpeta, puedes buscar por extension (busca la doc de "os")
def getImages(path):
    fileNames = os.listdir(path)
    images = []
    for f in fileNames:
        archivo = path + f
        estado = os.stat(archivo)
        tipo = estado.st_size
        if (tipo > 689670):
            img = readImg(archivo)
            images.append(img)
    return images

# guardar imagen (imagen referenciada, nombre) en formato png, lo puedes cambiar
def saveImg(img, name):
    s = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(name + '.png',s)


#####
# no importa todo esto :v
#####
def readImg(path):
    im = cv2.imread(path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    return im

def getBrightness(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    avg = ((np.sum(img[:,:,2])/(img.shape[0]*img.shape[1]))/255)*100
    print('Brightness %:',avg, sep=' ')
    img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
    return avg

def showImg(img,t=5):
    plt.figure(figsize = (t,t))
    plt.imshow(img)
    plt.axis('off')
    plt.show()
    print(img.shape)
    getBrightness(img)

def sumCh(img):
    SR = np.sum(img[:,:,0])
    SG = np.sum(img[:,:,1])
    SB = np.sum(img[:,:,2])
    return SR,SG,SB

def constraint(img):
    k = 1
    if (getBrightness(img) < 21.0):
        k = 1.5
    else:
        imgC = img.copy()
        while (getBrightness(imgC) > 21.0 and k > 0.80):
            imgC = img.copy()
            k -= 0.1
            imgC = updImg(imgC,k)
    print(k)
    return k

def calcGamma(img):
    g = 1.0
    imgC = img.copy()
    while (getBrightness(imgC) < 79 and g < 1.9):
        imgC = img.copy()
        g += 0.1
        imgC = adjust_gamma(imgC,g)
    print(g)
    return g

def updImg(img,k):
    imgUp = img.copy()
    imgUp[:,:,0] = img[:,:,0] * k
    imgUp[:,:,1] = img[:,:,1] * k
    imgUp[:,:,2] = img[:,:,2] * k
    return imgUp

def getAvgCh(img):
    SR, SG, SB = sumCh(img)
    y,x,_ = img.shape
    Ravg = SR/(y*x)
    Gavg = SG/(y*x)
    Bavg = SB/(y*x)
    print(Ravg,Gavg,Bavg, sep='\t')
    return Ravg,Gavg,Bavg

def shiftHist(imgUp):
    Ravg, Gavg, Bavg = getAvgCh(imgUp)
    diff1 = Ravg - Gavg
    diff2 = Bavg - Gavg
    print(diff1,diff2,sep='\t')
    img2 = imgUp.copy()
    img2[:,:,0] = img2[:,:,0] - diff1
    img2[:,:,2] = img2[:,:,2] - diff2
    return img2
    
def showColorHist(img,t = 5):
    color = ('r','g','b')
    plt.figure(figsize = (t,t))
    for i,col in enumerate(color):
        histr = cv2.calcHist([img],[i],None,[256],[0,256])
        plt.plot(histr,color = col)
        plt.xlim([0,256])
    plt.show()

def showBrightHist(img,t = 5):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    plt.figure(figsize = (t,t))
    plt.hist(img[:,:,2].ravel(),256,[0,256])
    plt.show()
    img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
    
def plotHists(img, t = 5):
    f = plt.figure(figsize = (t + 10,t))
    a = f.add_subplot(1,2,1)
    color = ('r','g','b')
    a.set_title('Color Histogram')
    for i,col in enumerate(color):
        histr = cv2.calcHist([img],[i],None,[256],[0,256])
        plt.plot(histr,color = col)
        plt.xlim([0,256])
    a = f.add_subplot(1,2,2)
    a.set_title('Brightness Histogram')
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    plt.hist(img[:,:,2].ravel(),256,[0,256])
    img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
    #plt.show()

def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
       for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)
