'''
@author:yk
基于直方图变换的风格迁移
修改os.chdir 输入python style.py xx.jpg(待变化的图片) xx.jpg(目标风格的图片)
'''
import cv2 as cv
import numpy as np
import random
import os
import matplotlib.pyplot as plt
import sys

os.chdir("C:\\Users\\m\\Desktop\\第三次作业")

def show(img,name="img"):               #显示图像
    cv.imshow(name,img)
    cv.waitKey(0)
    cv.destroyAllWindows()

def read(name):                        #读取图像
    return cv.imread(name+".bmp",0)

def hist_equal(img):                   #直方图均衡(求各个像素占比)
    M,N=img.shape
    s=np.zeros([256,1])
    for j in range(M):                #遍历每个像素的像素值
        for k in range(N):
            s[img[j][k]]+=1           #对应位置+1
    for i in range(1,256):
        s[i]=s[i-1]+s[i]      #累计求和          
    s=s/(M*N)
    return s

def hist_match(src,dst):               #直方图匹配
    M1,N1=src.shape
    M2,N2=dst.shape
    
    s=hist_equal(src)                 #src的sk
    z=hist_equal(dst)                 #dst的zk
    
    g=np.zeros([256])                 #初始化g函数
    index=0
    for i in range(256):             #寻找sk与zk最接近的一个数，返回下标作为索引值
        mins=1000
        for j in range(256):
            k=abs(s[i]-z[j])
            if k < mins:
                mins=k
                index=j
        g[i]=index
    return g

def img_trans(img,g):               #根据g函数，求出原图像关于g函数的转换，返回增强的图片
    M,N=img.shape
    dst=np.zeros(img.shape,dtype=np.uint8)
    for i in range(M):
        for j in range(N):
            dst[i][j]=g[img[i][j]]
    return dst

def img_enhance(img1,img2):    #绘制增强后的图以及其对应的直方图
    g=hist_match(img1,img2)
    dst=img_trans(img1,g)
    hist=cv.calcHist([dst],[0],None,[256],[0,256])
    plt.plot(hist)
    plt.ylim([0,10000])
    plt.clf()
    return dst
if __name__ =="__main__":
    name1=sys.argv[1]
    name2=sys.argv[2]
    orig1=cv.imread(name1)
    orig2=cv.imread(name2)
    b1,g1,r1=cv.split(orig1)
    b2,g2,r2=cv.split(orig2)
    dst1=img_enhance(b1,b2)
    dst2=img_enhance(g1,g2)
    dst3=img_enhance(r1,r2)
    dst=cv.merge([dst1,dst2,dst3])
    show(dst)
