'''
@author:yk7333 
 last modified:2021-4-22  
 language:python
'''

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os

os.chdir("C:\\Users\\m\\Desktop\\数字图像\\第6次作业")

def show(img,name="img"):
    cv.imshow(name,img)
    cv.waitKey(0)
    cv.destroyAllWindows()
def Noise(img,method="Gaussion",u=0,sigma=1):   #生成噪声
    if method=="Gaussion":
        noise=np.random.normal(u,sigma,img.shape).astype(int)
        image=np.zeros(img.shape,dtype=np.uint16)
        image=noise+img
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if image[i][j]>255:      #控制最大最小值为255与0
                    image[i][j]=255
                if image[i][j]<0:
                    image[i][j]=0

        image=image.astype(np.uint8)      #转化为uint8图像格式
        return image
    if method=="Salt" or method==1:
        img_copy=img.copy()
        mask=np.random.choice((0,1,2),size=img.shape,p=[0.8,0.1,0.1])
        img_copy[mask==1]=255
        img_copy[mask==2]=0
        return img_copy
    
def BorderProc(img,size):                              #外围补零操作，共size//2圈
    M,N=img.shape
    arr_x=np.zeros((M,size//2))
    arr_y=np.zeros((size//2,N+size-1))
    img=np.hstack([arr_x,img,arr_x])
    img=np.vstack([arr_y,img,arr_y])
    return img

def Calculate(img,size,method,i,j,Q):             #计算i,j点滤波之后的值
    
    arr=np.zeros((size,size))                         #arr记录img[i][j]附近待进行操作的元素
    i+=size//2;j+=size//2                            #因为外围增加了size//2圈0，因此在进行计算时，横纵轴均加size//2以定位到第一个非零元素
    for x in range(-size//2,size//2+1,1):            #从-size/2到size/2,依次在i,j处附近进行操作
        for y in range(-size//2,size//2+1,1):
            arr[x+size//2][y+size//2]=img[i+x][j+y]
    if method==0:                                   #算数均值滤波
        return np.mean(arr)
    if method==1:                                   #几何均值滤波
        result=1
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                result*=arr[i][j]
        
        return result**(1/(size**2))
    if method==2:                                   #谐波均值滤波
        result=0
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                result+=1/(arr[i][j]+0.0001)
        return size**2/result
    if method==3:                                   #逆谐波均值滤波
        result_up=0
        result_down=0
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                result_up+=(arr[i][j]+0.0001)**(Q+1)
                result_down+=(arr[i][j]+0.0001)**Q
        return result_up//result_down
    if method==4:
        return np.median(arr)

def Blur(img,size,method=0,Q=1):        #集合均值滤波操作
    M,N=img.shape
    dst=np.zeros(img.shape,dtype=np.uint8)
    img=BorderProc(img,size)
    for i in range(M):
        for j in range(N):
            dst[i][j]=Calculate(img,size,method,i,j,Q)
    return dst

def H_generate(shape,a,b,T):
    h,w=shape 
    H=np.zeros(shape)
    for i in range(h):
        for j in range(w):
            H[i][j]=T/(np.pi*i*a+j*b+0.00001)*np.sin(np.pi*(i*a+j*b))*np.exp(-1j*np.pi*(i*a+j*b))

    return H

def motion_blur(img,degree=50,angle=90):    #模糊处理
    M = cv.getRotationMatrix2D((degree/2,degree/2),angle,1)
    motion_blur_kernel = np.diag(np.ones(degree))
    motion_blur_kernel = cv.warpAffine(motion_blur_kernel, M, (degree,degree))
    motion_blur_kernel = motion_blur_kernel / degree
    result = cv.filter2D(img, -1, motion_blur_kernel)
    H=motion_blur_kernel
    cv.normalize(result,result,0,255, cv.NORM_MINMAX)
    result = np.array(result, dtype=np.uint8)
    return H,result

def H_back(H,K):
    M,N=H.shape
    H_b=np.zeros(H.shape)
    for i in range(M):
        for j in range(N):
             H_b[i][j]=1/H[i][j]*(np.abs(H[i][j]**2)/(K+np.abs(H[i][j]**2)))
    H_b[0][0]=0
    H_b[0][0]=np.max(H_b)           #第一个值为nan无穷大不便于处理 将其设置为其他值中的最大值
    return H_b

if __name__=="__main__":
    img=cv.imread("lena.bmp",0)
    h,w=img.shape
    image=Noise(img,u=0,sigma=20)      #高斯噪声
    #image=Noise(img,method=1)         #椒盐噪声
    cv.imwrite("noise.bmp",image)
    for i in range(5):
        result=Blur(image,size=3,method=i)       #method=0:算数均值 1:几何均值 2:谐波均值 3：逆谐波均值 4：中值滤波
        cv.imwrite("result_%d_0.bmp"%i,result)
        #cv.imwrite("result_%d_1.bmp"%i,result)

    # f = np.fft.fft2(img)
    # fshift = np.fft.fftshift(f) 

    # H=H_generate(f.shape,0.01,0.01,1)   #运动模糊（效果不好，改用matlab)
    # H_b=H_back(H,1)
    # fshift*=H
    # f1shift = np.fft.ifftshift(fshift)  # 逆变换
    # img_back = np.fft.ifft2(f1shift)
    # img_back = np.abs(img_back)
    # img_back= (img_back/np.max(img_back)*255).astype(np.uint8)
    # cv.imwrite("1.bmp",img_back)
