
import cv2
import numpy as np
import glob

# 设置寻找亚像素角点的参数，采用的停止准则是最大循环次数30和最大误差容限0.001
criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001)

# 获取标定板角点的位置
objp = np.zeros((6 * 9, 3), np.float32)
objp[:, :2] = np.mgrid[0:6, 0:9].T.reshape(-1, 2)  # 将世界坐标系建在标定板上，所有点的Z坐标全部为0，所以只需要赋值x和y

obj_points = []  # 存储3D点
img_points = []  # 存储2D点

images = glob.glob(r"C:\Users\kirin\Desktop\zzy_cam_calibration\camera_img2\*.jpg")
for fname in images:
    img = cv2.imread(fname)
    cv2.namedWindow('img')
    cv2.imshow('img',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    size = gray.shape[::-1]
    ret, corners = cv2.findChessboardCorners(gray, (6, 9), None)
    #print(ret)

    if ret:

        obj_points.append(objp)

        corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)  # 在原角点的基础上寻找亚像素角点
        #print(corners2)
        if [corners2]:
            img_points.append(corners2)
        else:
            img_points.append(corners)

        cv2.drawChessboardCorners(img, (8, 6), corners, ret)  # 记住，OpenCV的绘制函数一般无返回值
        cv2.imshow('img', img)
        cv2.waitKey(0)

print(len(img_points))
cv2.destroyAllWindows()

# 标定
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, size, None, None)

print("ret:", ret)
print("mtx(内参数矩阵):\n", mtx) # 内参数矩阵
print("dist(畸变系数):\n", dist)  # 畸变系数   distortion cofficients = (k_1,k_2,p_1,p_2,k_3)
print("rvecs(旋转向量):\n", rvecs)  # 旋转向量  # 外参数
print("tvecs(平移向量):\n", tvecs) # 平移向量  # 外参数

print("-----------------------------------------------------")
img = cv2.imread(images[2])
h, w = img.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))#显示更大范围的图片（正常重映射之后会删掉一部分图像）
print (newcameramtx)
dst = cv2.undistort(img,mtx,dist,None,newcameramtx)
x,y,w,h = roi
dst1 = dst[y:y+h,x:x+w]
cv2.imwrite('calibresult3.jpg', dst1)
print ("dst的大小为:", dst1.shape)

#计算误差
tot_error = 0
for i in range(len(obj_points)):
    imgpoints2, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(img_points[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    tot_error += error

print("total error: ", tot_error/len(obj_points))
cv2.destroyAllWindows()

#物体实测(未完成)
img = cv2.imread(r'C:\Users\kirin\Desktop\zzy_cam_calibration\1.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('img', gray)
cv2.waitKey(0)
ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
cv2.imshow('img', binary)
cv2.waitKey(0)
contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

draw_img0 = cv2.drawContours(img.copy(),contours,0,(0,255,255),3)
draw_img1 = cv2.drawContours(img.copy(),contours,1,(255,0,255),3)
draw_img2 = cv2.drawContours(img.copy(),contours,2,(255,255,0),3)
draw_img3 = cv2.drawContours(img.copy(), contours, -1, (0, 0, 255), 3)
cv2.imshow("draw_img0", draw_img0)
cv2.imshow("draw_img1", draw_img1)
cv2.imshow("draw_img2", draw_img2)
cv2.imshow("draw_img3", draw_img3)

cv2.waitKey(0)
cv2.destroyAllWindows()
print(contours)