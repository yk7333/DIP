import cv2 as cv
import numpy as np
import random
orig1 = cv.imread('Image A.jpg', 0)    
orig2 = cv.imread('Image B.jpg', 0)
img1=cv.resize(orig1,(500,500))                #修改大小，方便显示
img2=cv.resize(orig2,(500,500))

def show(img,name="img"):
    cv.namedWindow(name,0)
    cv.imshow(name,img)
    cv.waitKey(0)
    cv.destroyAllWindows()

sift = cv.xfeatures2d.SIFT_create()            #提取sift特征，keypoint
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)
matches = sorted(matches, key=lambda x: x.distance)
matches = bf.match(des1, des2)                 #BF蛮力匹配
random.shuffle(matches)                        #随机打乱顺序，选取7个点作为匹配点
mate= cv.drawMatches(img1, kp1, img2, kp2, matches[:7], None,flags=2)
src_pts = np.float32([ kp1[m[0].queryIdx].pt for m in good ]).reshape(-1,1,2)   #计算出原图和目标图每个点的坐标
dst_pts = np.float32([ kp2[m[0].trainIdx].pt for m in good ]).reshape(-1,1,2)
M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)                     #求出转化矩阵
    
rows1,cols1=img1.shape
dst=cv.warpAffine(img1,M[:2],(rows1,cols1))    #验证是否正确
print("转化矩阵T为:");print(M.T)               #输出转化矩阵
show(dst)
show(mate)

cv.imwrite("result.jpg",dst)
cv.imwrite("keypoint.jpg",mate)
