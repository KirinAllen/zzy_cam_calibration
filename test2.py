import numpy as np
import cv2
import glob

cbraw = 6
cbcol = 9

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((cbraw*cbcol,3), np.float32) 
objp[:,:2] = np.mgrid[0:cbraw,0:cbcol].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob('.\\camera_img2\\*.jpg')

for fname in images:
    img = cv2.imread(fname)
    #img = cv2.resize(img,None,fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #cv2.imshow('gray',gray)
    #cv2.waitKey()

    #寻找角点，存入corners，ret是找到角点的flag
    ret, corners = cv2.findChessboardCorners(gray, (cbraw,cbcol),None)
    #criteria:角点精准化迭代过程的终止条件
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        #执行亚像素级角点检测
        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (cbraw,cbcol), corners2,ret)
        cv2.imshow('img',img)
        cv2.waitKey()

#标定
# mtx 3*3内参数矩阵 dist 畸变系数 rvecs 旋转向量(外参数) 平移向量(外参数)
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
img = cv2.imread(".\camera_img\img_0.JPG")

#img = cv2.resize(img,None,fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)
h,w = img.shape[:2]
#显示更大范围的图片（正常重映射之后会删掉一部分图像）
newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
#纠正畸变
dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
#输出纠正畸变以后的图片
x,y,w,h = roi
dst = dst[y:y+h, x:x+w]
cv2.imwrite('calibresult.png',dst)
#输出矩阵参数
print("newcameramtx 外参:\n",newcameramtx)
#print("mtx 内参:\n",mtx)
print("dist 畸变值:\n",dist)
print ("newcameramtx旋转（向量）外参:\n",rvecs)
print ("dist平移（向量）外参:\n",tvecs)

#计算误差
tot_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    tot_error += error

print("total error: ", tot_error/len(objpoints))
cv2.destroyAllWindows()