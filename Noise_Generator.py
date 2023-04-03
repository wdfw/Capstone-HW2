import numpy.random 
import numpy as np
import cv2
import matplotlib.pyplot as plt

def showImg(img, position = 111, show = False) : #顯示灰階圖片
    plt.subplot(position) #使用subplot分割圖表
    plt.imshow(img,cmap='gray')  #將灰階圖送入buffer
    if show : plt.show() #是否直接顯示


def addNoisy(img, STD = 1) : #將圖片加上胡椒噪音, 並回傳新圖
    #標準高斯分佈之標準差, 在STD以外呈現雜訊, 否則為原圖
    h,w = img.shape[:2]
    
    #雜訊判斷對應像素, 判斷方式如上
    newImg = np.random.normal(0, 1, (h,w))
    
    for i in range(h) :
        for j in range(w) :
            #雜訊判斷, 正標準差外為白, 負標準差外為黑, 正負標準內為原像素
            newImg[i,j] = 255 if (newImg[i,j] > STD) else ( 0 if newImg[i,j] < -STD else img[i,j] )
    return newImg
#--------------------自行修改--------------------------
path = "/project_data/data_asset/doge.jpg" #自行修改路徑 !!!!!!!
img = cv2.imread(path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 轉換成灰階影像
#------------------------------------------------------

#原圖
print("Original")
showImg(img,111,show = True)

#加噪音 STD = 1
print("Add Noise STD = 1")
showImg(addNoisy(img, STD=1),111,show = True)
#加噪音 STD = 2
print("Add Noise STD = 2")
showImg(addNoisy(img, STD=2),111,show = True)
#加噪音 STD = 3
print("Add Noise STD = 3")
showImg(addNoisy(img, STD=3),111,show = True)
