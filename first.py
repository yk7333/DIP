'''
@author:自动化84杨恺    
last modified:2021-3-13 
language:python
'''
import cv2 as cv
import numpy as np
import time
orig=cv.imread("lena.bmp")
gray=cv.cvtColor(orig,cv.COLOR_BGR2GRAY)
def show(gray,name="image"):#显示函数
    cv.imshow(name,gray)
    cv.waitKey(0)
    cv.destroyAllWindows()
def gray_proc(gray,level):#图像灰度处理
    row,column=gray.shape[:]
    new=np.zeros(gray.shape)
    for i in range(row):
        for j in range(column):
            new[i][j]=gray[i][j]//(2**(8-level))   #转化为灰度级0到L-1
    maxnum=np.max(new)
    new=np.uint8(new/maxnum*255)                   #还原到0-255中，以便显示
    return new
def gray_disp(gray):  #图像灰度级显示
    tmp_img=None
    for level in range(8,0,-1):
        if level==4:         #图像拼接，4x2显示,到4时换行
            show_img=tmp_img
            tmp_img=None
            
        proc_img=gray_proc(gray,level)  #获取处理后的单个图片
        
        if tmp_img is None:              
            tmp_img=proc_img
        else:
            tmp_img=np.hstack([tmp_img,proc_img]) #图像横向拼接
    show_img=np.vstack([show_img,tmp_img])        #处理完后，纵向拼接为4*2
    show(show_img)
    cv.imwrite("gray_level.bmp",show_img)
def calculate(gray):           #均值与方差计算
    row=gray.shape[0]
    column=gray.shape[1]
    mean=0
    variance=0
    for i in range(row):
        for j in range(column):
            mean=mean+gray[i][j]
    mean=mean/(row*column)
    for i in range(row):
        for j in range(column):
            variance=variance+(gray[i][j]-mean)**2
    variance=variance/(row*column)
    return mean,variance
def reshape(gray,h,w,method):   #插值改变图片大小
    return cv.resize(gray,(h,w),interpolation=method)
def insert(gray):               #三种插值函数结果显示
    start=time.time()
    near=reshape(gray,2048,2048,cv.INTER_NEAREST)
    end=time.time()
    near_time=end-start
    start=time.time()
    linear=reshape(gray,2048,2048,cv.INTER_LINEAR)
    end=time.time()
    linear_time=end-start
    start=time.time()
    bicubic=reshape(gray,2048,2048,cv.INTER_CUBIC)
    end=time.time()
    bicubic_time=end-start
    print("\n临近法耗时:"+str(near_time)+"s\n"+"双线性插值耗时:"+str(linear_time)+"s\n"+"双三次插值法耗时:"+str(bicubic_time)+"s")
    cv.imwrite("near.bmp",near)
    cv.imwrite("linear.bmp",linear)
    cv.imwrite("bicubic.bmp",bicubic)
def process(gray,name):               #旋转和shear操作
    width,height=gray.shape[:]
    A1=cv.getRotationMatrix2D((width/2,height/2),30,0.7)
    W1=cv.warpAffine(gray,A1,(width,height))
    #show(W1)
    I1=reshape(W1,2048,2048,cv.INTER_NEAREST)
    I2=reshape(W1,2048,2048,cv.INTER_LINEAR)
    I3=reshape(W1,2048,2048,cv.INTER_CUBIC)
    cv.imwrite(name+" rotate nearest.bmp",I1)
    cv.imwrite(name+" rotate linear.bmp",I2)
    cv.imwrite(name+" rotate bicubic.bmp",I3)

    A2=np.array([[1,1.5,0],[0,1,0]])
    W2=cv.warpAffine(gray,A2,(int(width*2.5),int(height*1)))
    #show(W2)
    I1=reshape(W2,2048,2048,cv.INTER_NEAREST)
    I2=reshape(W2,2048,2048,cv.INTER_LINEAR)
    I3=reshape(W2,2048,2048,cv.INTER_CUBIC)
    cv.imwrite(name+" shear nearest.bmp",I1)
    cv.imwrite(name+" shear linear.bmp",I2)
    cv.imwrite(name+" shear bicubic.bmp",I3)
    
    
gray_disp(gray)               #第二问：灰度级8-1显示

mean,variance=calculate(gray) #第三问：均值与方差
print("mean:"+str(mean)+"\t\tvariance:"+str(variance))

insert(gray)                  #第四问：三种插值方式
process(gray,"lena")          #第五问：旋转、shear后进行插值操作
gray=cv.cvtColor(cv.imread("elain1.bmp"),cv.COLOR_BGR2GRAY)
process(gray,"elain")
print("done!")
