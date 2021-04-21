'''
@author:yk7333 
 last modified:2021-4-7  
 language:python
'''

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os


os.chdir("C:\\Users\\m\\Desktop\\第五次作业")
def read(name):
    return cv.imread(name,0)

#D0：截止频率
#shape:图像大小
#n为巴特沃斯滤波器阶数
#lowpass=True低通 False高通

def Filter(method,D0,shape,lowpass,n=2):
    M,N=shape
    center_x,center_y=M//2,N//2                        #中心点
    H=np.zeros((M,N),dtype=np.float32)                
    D=lambda x,y:((center_x-x)**2+(center_y-y)**2)**(1/2)#距离函数
    if method=="UM":
        for i in range(M):                                #生成unmask滤波器
            for j in range(N):
                if D(i,j)>D0:
                    H[i][j]=1 
    if method=="BW":
        for i in range(M):                                #生成巴特沃斯滤波器
            for j in range(N):
                H[i][j]=1/(1+(D(i,j)/D0)**(2*n))if lowpass is True else (1-1/(1+(D(i,j)/D0)**(2*n)))      
    if method=="GS":
        for i in range(M):                                #生成高斯沃斯滤波器
            for j in range(N):
                H[i][j]=np.exp(-D(i,j)**2/(2*D0**2))if lowpass is True else (1-np.exp(-D(i,j)**2/(2*D0**2)))
    if method=="LP":
        for i in range(M):                                #生成Laplacian滤波器
            for j in range(N):
                H[i][j]=4*(np.pi**2)*D(i,j)
        H/=np.max(H)
    return H

#D0：截止频率
#method：滤波方式
#lowpass：滤波方式选择
def Blur(img,D0,method="BW",lowpass=True):
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    plt.subplot(121),plt.imshow(img,'gray'),plt.title('original')
    amplitude1=np.sum(np.abs(fshift*fshift)) #原始图的功率
    
    H=Filter(method,D0,fshift.shape,lowpass)
    fshift=fshift*H
    amplitude2=np.sum(np.abs(fshift*fshift)) #滤波后的功率
    
    f1shift = np.fft.ifftshift(fshift)  # 逆变换
    img_back = np.fft.ifft2(f1shift)

    img_back = np.abs(img_back)      #出来的是复数，转化为模值大小
    plt.subplot(122),plt.imshow(img_back,'gray'),plt.title('dst')
    #cv.imwrite("{0}_{1}_lowpass.jpg".format(D0,method),img_back)
    cv.imwrite("{0}_{1}_highpass.jpg".format(D0,method),img_back)
    print("功率谱比为"+"%.4f" %(amplitude2/amplitude1))
if __name__ =="__main__":
    img = read("test4.tif") 
    Blur(img,1,"UM",lowpass=0)       #频域滤波操作
