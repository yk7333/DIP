import cv2 as cv
import numpy as np
import random
import os
import matplotlib.pyplot as plt

os.chdir("C:\\Users\\m\\Desktop\\第三次作业")
name=["citywall","citywall1","citywall2","elain","elain1","elain2","elain3","lena","lena1","lena2","lena4","woman","woman1","woman2"]
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
        #sk矩阵
    return s

def hist_match(src,dst):               #直方图匹配
    M1,N1=src.shape
    M2,N2=dst.shape
    
    s=hist_equal(src)                 #src的sk
    z=hist_equal(dst)                 #dst的zk
    
    g=np.zeros([256])                 #初始化g函数
    index=None
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
            dst[i][j]=round(g[img[i][j]])
    return dst

def img_enhance(img1,img2,name):    #绘制增强后的图以及其对应的直方图
    g=hist_match(img1,img2)
    dst=img_trans(img1,g)
    cv.imwrite(name+"_enhanced.bmp",dst)
    hist=cv.calcHist([dst],[0],None,[256],[0,256])
    plt.plot(hist)
    plt.ylim([0,10000])
    plt.savefig(name+"_enhanced_hist.jpg")
    plt.clf()
    
def img_hist(name):                 #绘制直方图以及均衡化之后的直方图
    for each in name:
        hist=cv.calcHist([read(each)],[0],None,[256],[0,256])
        plt.plot(hist)
        plt.ylim([0,10000])
        plt.savefig("{0}.jpg".format(each))
        plt.clf()

        equ=cv.equalizeHist(read(each))  #均衡化
        cv.imwrite("{0}_equ.bmp".format(each),equ)
        hist=cv.calcHist([equ],[0],None,[256],[0,256])
        plt.plot(hist)
        plt.ylim([0,10000])
        plt.savefig("{0}_equ_hist.jpg".format(each))
        plt.clf()
def img_local_hist(name):         #局部均衡化
    img=read(name)
    result=cv.createCLAHE(clipLimit=2,tileGridSize=(7,7)).apply(img)
    cv.imwrite(name+"_local.bmp",result)
    hist=cv.calcHist([img],[0],None,[256],[0,256])
    plt.plot(hist)
    plt.savefig(name+"_local_hist.jpg")
    plt.clf()
def img_divide(name,thresh):     #图像阈值分割
    img=read(name)
    return cv.threshold(img,thresh,255,cv.THRESH_BINARY)

img_hist(name[11:])
img_enhance(read(name[1]),read(name[0]),"citywall_new")
img_local_hist(name[7])
_,dst=img_divide(name[3],100)
#show(dst)
cv.imwrite("woman_div.jpg",dst)
