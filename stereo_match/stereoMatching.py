# -*- coding: utf-8 -*-
from cv2 import cv2
import numpy as np
import stereoconfig
import time
 
 
# 预处理
def preprocess(img1, img2):
    # 彩色图->灰度图
    if(img1.ndim == 3):
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)  # 通过OpenCV加载的图像通道顺序是BGR
    if(img2.ndim == 3):
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
 
    # 直方图均衡
    img1 = cv2.equalizeHist(img1)
    img2 = cv2.equalizeHist(img2)
 
    return img1, img2
 
 
# 消除畸变
def undistortion(image, camera_matrix, dist_coeff):
    undistortion_image = cv2.undistort(image, camera_matrix, dist_coeff)
 
    return undistortion_image
 
 
# 获取畸变校正和立体校正的映射变换矩阵、重投影矩阵
# @param：config是一个类，存储着双目标定的参数:config = stereoconfig.stereoCamera()
def getRectifyTransform(height, width, config):
    # 读取内参和外参
    left_K = config.cam_matrix_left
    right_K = config.cam_matrix_right
    left_distortion = config.distortion_l
    right_distortion = config.distortion_r
    R = config.R
    T = config.T
 
    # 计算校正变换
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(left_K, left_distortion, right_K, right_distortion, (width, height), R, T, alpha=0)
 
    map1x, map1y = cv2.initUndistortRectifyMap(left_K, left_distortion, R1, P1, (width, height), cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(right_K, right_distortion, R2, P2, (width, height), cv2.CV_32FC1)
 
    return map1x, map1y, map2x, map2y, Q
 
 
# 畸变校正和立体校正
def rectifyImage(image1, image2, map1x, map1y, map2x, map2y):
    rectifyed_img1 = cv2.remap(image1, map1x, map1y, cv2.INTER_AREA)
    rectifyed_img2 = cv2.remap(image2, map2x, map2y, cv2.INTER_AREA)
 
    return rectifyed_img1, rectifyed_img2
 
 
# 立体校正检验----画线
def draw_line(image1, image2):
    # 建立输出图像
    height = max(image1.shape[0], image2.shape[0])
    width = image1.shape[1] + image2.shape[1]
 
    output = np.zeros((height, width, 3), dtype=np.uint8)
    output[0:image1.shape[0], 0:image1.shape[1]] = image1
    output[0:image2.shape[0], image1.shape[1]:] = image2
 
    # 绘制等间距平行线
    line_interval = 50  # 直线间隔：50
    for k in range(height // line_interval):
        cv2.line(output, (0, line_interval * (k + 1)), (2 * width, line_interval * (k + 1)), (0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
 
    return output
 
 
# 视差计算
def stereoMatchSGBM(left_image, right_image, down_scale=False):
    # SGBM匹配参数设置
    if left_image.ndim == 2:
        img_channels = 1
    else:
        img_channels = 3
    blockSize = 11
    paraml = {'minDisparity': 0,
             'numDisparities': 48,
             'blockSize': blockSize,
             'P1': 8 * img_channels * blockSize ** 2,
             'P2': 32 * img_channels * blockSize ** 2,
             'disp12MaxDiff': 1,
             'preFilterCap': 63,
             'uniquenessRatio': 15,
             'speckleWindowSize': 100,
             'speckleRange': 1,
             'mode': cv2.STEREO_SGBM_MODE_SGBM_3WAY
             }
 
    # 构建SGBM对象
    left_matcher = cv2.StereoSGBM_create(**paraml)
    paramr = paraml
    paramr['minDisparity'] = -paraml['numDisparities']
    right_matcher = cv2.StereoSGBM_create(**paramr)
 
    # 计算视差图
    size = (left_image.shape[1], left_image.shape[0])
    if down_scale == False:
        disparity_left = left_matcher.compute(left_image, right_image)
        disparity_right = right_matcher.compute(right_image, left_image)
 
    else:
        left_image_down = cv2.pyrDown(left_image)
        right_image_down = cv2.pyrDown(right_image)
        factor = left_image.shape[1] / left_image_down.shape[1]
 
        disparity_left_half = left_matcher.compute(left_image_down, right_image_down)
        disparity_right_half = right_matcher.compute(right_image_down, left_image_down)
        disparity_left = cv2.resize(disparity_left_half, size, interpolation=cv2.INTER_AREA)
        disparity_right = cv2.resize(disparity_right_half, size, interpolation=cv2.INTER_AREA)
        disparity_left = factor * disparity_left
        disparity_right = factor * disparity_right
 
    # 真实视差（因为SGBM算法得到的视差是×16的）
    trueDisp_left = disparity_left.astype(np.float32) / 16.
    trueDisp_right = disparity_right.astype(np.float32) / 16.
 
    return trueDisp_left, trueDisp_right

def computeDepth(disparity,matrix_Q):
    threeD = cv2.reprojectImageTo3D(disparity.astype(np.float32) / 16., matrix_Q)
    return threeD


number = "002"
Type = "LaRb"  #_real_A  _fake_B  LbRb
Format = ".jpg"
if __name__ == '__main__':
    start = time.time()
    # 读取MiddleBurry数据集的图片
    # iml = cv2.imread('test_2/left_' + number + Format)  # 左图
    # imr = cv2.imread('test_2/right_' + number + Format)  # 右图
    # iml = cv2.imread('test_3/536.162358_real_A.png')  # 左图 536.162358
    # imr = cv2.imread('test_3/536.162351_real_A.png')  # 右图 536.162358
    iml = cv2.imread('/Users/lvpengkai/Desktop/毕业设计/桌面文件/images2/71.585022_fake_B.png')
    imr = cv2.imread('/Users/lvpengkai/Desktop/毕业设计/桌面文件/images2/71.585038.jpg')
    # iml = cv2.imread('images_1/left_28.jpg')
    # imr = cv2.imread('/Users/lvpengkai/Desktop/right_28_fake_B.png')
    height, width = iml.shape[0:2]
    counter = 0
    # 读取相机内参和外参
    config = stereoconfig.stereoCamera()
 
    # 立体校正
    map1x, map1y, map2x, map2y, Q = getRectifyTransform(height, width, config)  # 获取用于畸变校正和立体校正的映射矩阵以及用于计算像素空间坐标的重投影矩阵
    iml_rectified, imr_rectified = rectifyImage(iml, imr, map1x, map1y, map2x, map2y)
    # print(Q)
 
    # 绘制等间距平行线，检查立体校正的效果
    line = draw_line(iml_rectified, imr_rectified)
    cv2.imwrite('test_4/检验_02.png', line)
    # cv2.imshow('检验_02', line)
 
    # 立体匹配
    iml_, imr_ = preprocess(iml_rectified, imr_rectified )  # 预处理，一般可以削弱光照不均的影响，不做也可以
    disp, _ = stereoMatchSGBM(iml_, imr_, True)
    # Depth = computeDepth(disp, Q)
    # Depth = cv2.reprojectImageTo3D(disp, Q)
    # a=271
    # b=262
    i=0.50*480
    j=0.17*640
    int_i = int(i)
    int_j = int(j)
    h = 1.00*480
    w = 0.34*640
    maxdisp = 0
    sumdisp = 0
    counter = 0
    # methor_1
    for j in range( int(int_j - w//2), int(int_j + w//2) ):
        if disp[int(i)][j] > maxdisp:
            maxdisp = disp[int(i)][j]
    # for i in range( int(int_i - h//2), int(int_i + h//2) ):
    #     for j in range( int(int_j - w//2), int(int_j + w//2)):
    #         if disp[i][j] > 0:
    #             counter+=1
    #             sumdisp += disp[i][j]
    # maxdisp=sumdisp/counter
    # Depth = (99.3638 * 640.904567)/maxdisp 
    # Depth = cv2.reprojectImageTo3D(disp, Q)
    end = time.time()
    print((end - start),"s")
    # print(str(abs(Depth)[b][a][2])+ " mm") #[y][x][2] 宽、长
    # print(Depth, " mm")
    
    cv2.imwrite('test_4/' + number + Type + 'disp1.png', disp)
    # cv2.imwrite('test_3/' + number + Type + 'rectified_l.png', iml_rectified)
    # cv2.imwrite('test_3/' + number + Type + 'rectified_r.png', imr_rectified)
    disp_ = np.array(disp,np.uint8)
    fakeColorDepth = cv2.applyColorMap(disp_, cv2.COLORMAP_JET)
    cv2.imwrite('test_4/' + number + Type + 'fcolor1.png', fakeColorDepth)

