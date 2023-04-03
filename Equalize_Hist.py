import numpy as np
import cv2
import matplotlib.pyplot as plt

def pixelCount(img) : #算出灰階像素的數量
    h,w = img.shape[:2]
    res = np.zeros(256).astype(np.uint32) #記錄灰階像素 0 - 255 的數量
    for i in range(h) : #統計所有像素
        for j in range(w) :
            res[img[i,j]] += 1
    return res

def drawGrayHist(img, position = 111, show = False) : #以直方圖畫出灰階像素分佈
    pixel = pixelCount(img) #算出灰階像素分佈
    plt.subplot(position) #使用subplot分割圖表
    plt.bar(range(0,256),pixel) #將常條圖送入buffer
    if show : plt.show() #是否直接顯示
        
def showImg(img, position = 111, show = False) : #顯示灰階圖片
    plt.subplot(position) #使用subplot分割圖表
    plt.imshow(img,cmap='gray')  #將灰階圖送入buffer
    if show : plt.show() #是否直接顯示
    
def myEqualizeHist(img) : #手刻直方等化
    h,w = img.shape[:2]
    newImg = np.zeros((h,w)) #生出新圖
    
    distribute = pixelCount(img) #先算出灰階分佈(PDF)
    #將灰階分佈歸一化
    total = sum(distribute)
    #歸一化
    pdf = distribute/total
    
    #疊加創建CDF
    cdf = [pdf[0]]
    
    for i in range(1,256) : cdf.append(cdf[-1] + pdf[i])
    #以映射方式將原圖線性轉換    
    mappingTable = [round(i*255) for i in cdf]
    for i in range(h) :
        for j in range(w) :
            newImg[i,j] = mappingTable[img[i,j]]
    return newImg.astype(np.uint8)

#--------------------自行修改--------------------------
path = "/project_data/data_asset/doge.jpg" #自行修改路徑 !!!!!!!
img = cv2.imread(path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 轉換成灰階影像
#-----------------------------------------------------

#原圖部分
print("Original")
showImg(img,121)
drawGrayHist(img,122,show = True)

#CV2部分
print("CV2 Equalize Hist")
CV2equal_img = cv2.equalizeHist(img)
showImg(CV2equal_img,121)
drawGrayHist(img,122,show = True)

#手刻部分
print("Own Equalize Hist")
Myequal_img = myEqualizeHist(img)
showImg(Myequal_img,121)
drawGrayHist(img,122,show = True)
