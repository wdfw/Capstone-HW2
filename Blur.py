import numpy as np
import cv2
import matplotlib.pyplot as plt

def showImg(img, position = 111, show = False) : #顯示灰階圖片
    plt.subplot(position) #使用subplot分割圖表
    plt.imshow(img,cmap='gray')  #將灰階圖送入buffer
    if show : plt.show() #是否直接顯示

def MyMedianBlur(img,kernel) : #中值濾波
    lengthL = (kernel-1) // 2 #右下方長度
    lengthR = kernel - lengthL - 1 #左上方長度
    
    h,w = img.shape[:2]
    newImg = np.zeros((h,w))
    
    for i in range(h) :
        for j in range(w) :
            
            upper =  max(i-lengthL,0) #上邊界
            bottom = min(i+lengthR,h-1) #下邊界
            
            left =  max(j-lengthL,0) #左邊界
            right = min(j+lengthR,w-1) #右邊界
            #透過np.median 找邊界內像素的中間值
            newImg[i,j] = np.median(img[upper:bottom+1,left:right+1])
    return newImg.astype(np.uint8)

def MyMeanBlur(img,kernel) : #均值濾波
    lengthL = (kernel-1) // 2 #右下方長度
    lengthR = kernel - lengthL - 1 #左上方長度
    
    h,w = img.shape[:2]
    newImg = np.zeros((h,w))

    for i in range(h) :
        for j in range(w) :
            
            upper =  max(i-lengthL,0) #上邊界
            bottom = min(i+lengthR,h-1) #下邊界
            
            left =  max(j-lengthL,0) #左邊界
            right = min(j+lengthR,w-1) #右邊界
            sizeCol = bottom - upper + 1 #上下邊界的大小
            sizeRow = right - left + 1 #左右邊界的大小
            #透過np.sum 算出邊界範圍內的總合, 並平均
            newImg[i,j] = np.sum(img[upper:bottom+1,left:right+1]) // (sizeCol*sizeRow)
    return newImg.astype(np.uint8)

#--------------------自行修改--------------------------
path = "/project_data/data_asset/doge.jpg" #自行修改路徑 !!!!!!!
img = cv2.imread(path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 轉換成灰階影像
#------------------------------------------------------

#原圖部分
print("Original")
showImg(img)
showImg(img,111,show=True)


#CV2中值濾波 與 手刻中值濾波
print("cv2.medianBlur VS MyMedianBlur")
CV2blurImg = cv2.medianBlur(img, 5)
MyBlurImg = MyMedianBlur(img,5)

showImg(CV2blurImg,121)
showImg(MyBlurImg,122,show=True)


#手刻均值濾波
print("Own meanBlur")
MyBlurImg = MyMeanBlur(img,5)
showImg(MyBlurImg,111,show=True)
