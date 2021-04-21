'''
@author:yk7333 
 last modified:2021-4-7  
 language:python
'''
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os

def show(img,name="img"):
    cv.imshow(name,img)
    cv.waitKey(0)
    cv.destroyAllWindows()
    
def read(path):
    return cv.imread(path,0)

def save(name,src):
    return cv.imwrite(name,src)

def p(x,y,sigma):                                      #高斯数值生成
    return 1/(2*np.pi*sigma**2)*np.exp(-(x**2+y**2)/(2*sigma**2))
def norm(arr):                                         #核归一化
    sumary=np.sum(arr)
    return arr/sumary
    
def Gaussion(size,sigma):                              #生成高斯核
    gaussion=np.zeros((size,size))
    center_x,center_y=size//2,size//2
    for i in range(size):
        for j in range(size):
            gaussion[i][j]=p(i-center_x,j-center_y,sigma)
    gaussion=norm(gaussion)
    return gaussion

def BorderProc(img,size):                              #外围补零操作，共size//2圈
    M,N=img.shape
    arr_x=np.zeros((M,size//2))
    arr_y=np.zeros((size//2,N+size-1))
    img=np.hstack([arr_x,img,arr_x])
    img=np.vstack([arr_y,img,arr_y])
    return img

def Calculate(img,size,method,i,j,sigma):             #计算i,j点滤波之后的值
    
    arr=np.zeros((size,size))                         #arr记录img[i][j]附近待进行操作的元素
    i+=size//2;j+=size//2                            #因为外围增加了size//2圈0，因此在进行计算时，横纵轴均加size//2以定位到第一个非零元素
    for x in range(-size//2,size//2+1,1):            #从-size/2到size/2,依次在i,j处附近进行操作
        for y in range(-size//2,size//2+1,1):
            arr[x+size//2][y+size//2]=img[i+x][j+y]
    if method=="Gaussion":                           #高斯滤波         
        blur=Gaussion(size,sigma)
        return np.sum(arr*blur)
    if method=="Median":                             #中值滤波
        return np.median(arr)
    
def Blur(img,size,method="Gaussion",sigma=1):        #滤波操作
    M,N=img.shape
    dst=np.zeros(img.shape,dtype=np.uint8)
    img=BorderProc(img,size)
    for i in range(M):
        for j in range(N):
            dst[i][j]=Calculate(img,size,method,i,j,sigma)
    return dst
     
if __name__ == "__main__":
    os.chdir("C:\\Users\\m\\Desktop\\第四次作业")
    
    for i in range(3,8,2):                      #3,5,7
        img=read("test2.tif")                   #第一问

        gaussion=Blur(img,i,"Gaussion")
        median=Blur(img,i,"Median")
        save("gaussion2{0}x{1}.jpg".format(i,i),gaussion)
        save("medium2{0}x{1}.jpg".format(i,i),median)

    for i in range(3,8,2):
        print(Gaussion(i,1.5))                       #第二问
        print("\n")

    img3=read("test3_corrupt.pgm")
    img4=read("test4 copy.bmp")
#unshape masking
    img3_blur=Blur(img3,5,sigma=1)                     #采用5x5高斯滤波进行模糊处理
    img4_blur=Blur(img4,5,sigma=1) 
    mask3=img3-img3_blur
    mask4=img4-img4_blur
    save("img3_unmask.jpg",mask3)
    save("img4_unmask.jpg",mask4)
#Sobel edge detector
    sobelx=cv.Sobel(img3,cv.CV_64F,0,1,ksize=3)
    sobelx=cv.convertScaleAbs(sobelx)
    sobely=cv.Sobel(img3,cv.CV_64F,1,0,ksize=3)
    sobely=cv.convertScaleAbs(sobely)
    sobelxy=cv.addWeighted(sobelx,0.5,sobely,0.5,0) 
    save("img3_sobel.jpg",sobelxy)
    sobelx=cv.Sobel(img4,cv.CV_64F,0,1,ksize=3)
    sobelx=cv.convertScaleAbs(sobelx)
    sobely=cv.Sobel(img4,cv.CV_64F,1,0,ksize=3)
    sobely=cv.convertScaleAbs(sobely)
    sobelxy=cv.addWeighted(sobelx,0.5,sobely,0.5,0)
    save("img4_sobel.jpg",sobelxy)
#laplace edge detection
    laplacian = cv.Laplacian(img3,cv.CV_64F)
    laplacian = cv.convertScaleAbs(laplacian) 
    save("img3_lap.jpg",laplacian)
    laplacian = cv.Laplacian(img4,cv.CV_64F)
    laplacian = cv.convertScaleAbs(laplacian) 
    save("img4_lap.jpg",laplacian)
#canny algorithm
    canny=cv.Canny(img3,50,80)
    save("img3_canny.jpg",canny)
    canny=cv.Canny(img4,50,80)
    save("img4_canny.jpg",canny)
