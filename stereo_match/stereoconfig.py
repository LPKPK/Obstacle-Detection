import numpy as np
 
 
####################仅仅是一个示例###################################
 
 
# 双目相机参数
class stereoCamera(object):
    def __init__(self):
        # 左相机内参
        self.cam_matrix_left = np.array([[  587.7023 , -0.4290 , 322.3464],
                                [0,  586.7238,  238.0438],
                                [0., 0., 1.]])
        # 右相机内参
        self.cam_matrix_right = np.array([[  586.9051,    -0.7486, 350.1402],
                               [        0,  585.3174,  229.5179],
                               [         0,         0,    1.]])
        # 左右相机畸变系数:[k1, k2, p1, p2, k3]
        self.distortion_l = np.array([[0.0157,	-0.0050, -0.0016, 0.0021, 0]])
        self.distortion_r = np.array([[0.0274,	-0.0545, -0.0015 ,0.0050, 0]])
 
        # 旋转矩阵
        self.R = np.array([
    [ 0.9997,   -0.0090,   0.0207],
    [ 0.0088,    0.9999,   0.0091],
    [-0.0207,   -0.0089,   0.9997],])
        # 平移矩阵
        self.T = np.array([-99.3638	, -0.5474 , -2.3216])
        # 焦距  2574.91511632263  	2522.62324015240
        self.focal_length = 640.904567 # 默认值，一般取立体校正后的重投影矩阵Q中的 Q[2,3]

        # 基线距离
        self.baseline = 99.3638  # 单位：mm， 为平移向量的第一个参数（取绝对值）

