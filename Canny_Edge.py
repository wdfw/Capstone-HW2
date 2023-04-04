import numpy as np
import cv2
import matplotlib.pyplot as plt
import math 

def showImg(img, position = 111, show = False) : #顯示灰階圖片
    plt.subplot(position) #使用subplot分割圖表
    plt.imshow(img,cmap='gray')  #將灰階圖送入buffer
    if show : plt.show() #是否直接顯示
    
def sobelKernel(direction) : #回傳sobel kernek | direction = 0 回傳X軸的核 | direction=90 回傳Y軸的核 
    res = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    if direction == 0 : return res
    elif direction == 90 : return res.T
    
def normalKernel(k, ro) : #回傳高斯核 ro代表高斯核的標準差
    kernel = np.zeros((2+k,2+k))
    constant = 2*math.pi*(ro*ro)
    for i in range(2+k) :
        for j in range(2+k) :
            e = math.exp( -( (i-k)*(i-k) + (j-k)*(j-k) )/(2*ro*ro)  ) #透過高斯公式設計核的參數
            kernel[i,j] = e / constant 
    return kernel/np.sum(kernel) #回傳歸一化後的高斯核

#角度常數
#TAN22_5 = math.pi / 8
#TAN67_5 = math.pi / 8 * 3
TAN22_5 = 0.414
TAN67_5 = 2.414
def angel(p) : #將梯度轉換為方向 分為 0垂直 1斜下 2平行 3斜上
    global TAN22_5, TAN67_5 
    #角度判斷 p要輸入角度 !!
    if TAN67_5 > p >= TAN22_5 : return 3 #22.5 ~ 67.5 方向為 斜上 /
    elif TAN22_5 > p >= -TAN22_5 : return 2 # 22.5 ~ -22.5 方向為 平行 -
    elif -TAN22_5 > p >= -TAN67_5 : return 1 # -22.5 ~ -67.5 方向為 斜下 \ 
    else : return 0 # 其餘 方向為 |
    
def sobel(img,dx,dy) :#sobel運算子 dx==1使用平行核 dy==1使用垂直核
    if dx : kernel = sobelKernel(0) #決定sobel核
    elif dy : kernel = sobelKernel(90) 
    h,w = img.shape[:2]
    G = np.zeros((h,w))
    #進行捲積 最邊界為0
    for i in range(h-2) :
        for j in range(w-2) : 
            #透過圖與核的相乘並相加得到結果
            res = np.sum(np.multiply(kernel,img[i:i+3,j:j+3])) 
            #將結果限制於0-255
            #res = max(0,res)
            #res = min(255,res)
            G[i+1,j+1] = res
    return G

def MyNoiseRemoval(img,kernel) : #高斯濾波
    if kernel % 2 == 0 : raise ValueError("kernel length must be odd") #高斯濾波的kernel一定是奇數
        
    lengthL = (kernel-1) // 2 #右下方長度
    lengthR = kernel - lengthL - 1 #左上方長度
    
    h,w = img.shape[:2]
    newImg = np.zeros((h,w))
    mask = normalKernel(kernel-2,1.5) #取得kernel, 實驗使用固定ro = 1.5
    for i in range(h) :
        for j in range(w) :
            upper =  max(i-lengthL,0) #上邊界
            bottom = min(i+lengthR,h-1) #下邊界
            
            left =  max(j-lengthL,0) #左邊界
            right = min(j+lengthR,w-1) #右邊界

            #針對邊緣進行kernel位置的調整
            mx =  kernel // 2 #kernel X軸的中間值
            my =  kernel // 2 #kernel Y軸的中間值
            ku = - i + upper + mx  #kernel 調整的上邊界
            kb = bottom - i + mx #kernel 調整的下邊界
            kl =  - j + left + my #kernel 調整的左邊界
            kr = right - j + my #kernel 調整的右邊界
            
            res = np.sum(np.multiply(mask[ku:kb+1,kl:kr+1],img[upper:bottom+1,left:right+1])) #高斯核與圖的捲積
            normal = np.sum(mask[ku:kb+1,kl:kr+1]) #高斯核使用的範圍
            newImg[i,j] = res/normal #將結果歸一化
    return newImg

def Direction(T) : #將T由角度轉為方向
    for i in range(len(T)) :
        for j in range(len(T[i])) :
            T[i,j] = angel(T[i,j]) #判斷此點的方向
            
def NonMaximum(G,D) : #與鄰居比對判斷是否最大值
    h,w = D.shape[:2]
    newImg = np.zeros((h,w))
    for i in range(1,len(D)-1) : 
        for j in range(1,len(D[i])-1) :
            if D[i,j] == 0 : v1,v2 = G[i-1,j],G[i+1,j] #當方向垂直 找垂直鄰點
            elif D[i,j] == 3 : v1,v2 = G[i+1,j+1],G[i-1,j-1] #當方向斜下 找斜下鄰點
            elif D[i,j] == 2 : v1,v2 = G[i,j+1],G[i,j-1] #當方向平行 找平行鄰點
            elif D[i,j] == 1 : v1,v2 = G[i-1,j+1],G[i+1,j-1] #當方向斜上 找斜上鄰點
            if(not G[i,j] <= v1 and not G[i,j] <= v2) : newImg[i,j] = G[i,j] #如果是極大點 就給此像素值
    return newImg
            

def Threadhold(G,lower,upper) : #臨界值判定 
    #當value > upper : 為邊界 | value < lower : 非邊界 | lower < value < upper 可能邊界
    for i in range(len(G)) : 
        for j in range(len(G[i])) :
            if G[i,j] > upper : G[i,j] = 255 # value > upper : 為邊界
            elif G[i,j] < lower : G[i,j] = 0 # value < lower : 非邊界
            else : #lower < value < upper 可能邊界
                G[i,j] = 0 
                for k in range(9) : #進行鄰界的8格判斷使否為edge
                    if G[i-1+k//3 ,j-1+k%3] > upper :
                        G[i,j] = 255 
                        break

                
def MyCanny(img, TL, TH) : #Canny Edge Detection
    if len(img.shape) >= 3 : #如果輸入彩色影像 轉灰階
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 轉換成灰階影像
        
    img = img.astype(np.float64) #防止高斯濾波造成的丟失
    img = MyNoiseRemoval(img,5)
    
    Gx = sobel(img,1,0)
    Gy = sobel(img,0,1)
   
    G = ((Gx**2 + Gy**2)**0.5).astype(np.uint8)
    
    D = Gy/(Gx+10e-6)#np.arctan(Gy/(Gx+10e-6)) #設定10e-6防止除0
    
    #將數值轉為角度 再轉為方向
    Direction(D)
    #非極大值抑制
    G = NonMaximum(G ,D) 
    #雙閥值判定
    Threadhold(G,TL,TH) 
    #回傳結果
    return G

if __name__ == "__main__" :
    
    #--------------------自行修改--------------------------
    path = "/project_data/data_asset/doge.jpg" #自行修改路徑 !!!!!!!
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
    TL,TH = 30,60 #高低閥值設定
    #-----------------------------------------------------
    
    G = MyCanny(img,TL,TH) #手刻 Canny(image, Threadhold_low, Threadhold_high)
    cvG = cv2.Canny(MyNoiseRemoval(img,5).astype(np.uint8), TL, TH) #輸入濾波後的圖給CV2 Canny
    
    #原圖部分
    print("Original")
    showImg(img,111,show = True)
    
    #Canny CV2部分
    print("CV2")
    showImg(cvG,111,show = True)
    
    #Canny 手刻部分
    print("Own")
    showImg(G,111,show = True)
