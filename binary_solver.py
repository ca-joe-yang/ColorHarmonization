import cv2
import numpy as np
import sys
import util
import itertools

import importlib
importlib.reload(util)

image_filename = sys.argv[1]
color_image = cv2.imread(image_filename, cv2.IMREAD_COLOR)



HSV_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)

I = HSV_image
height = I.shape[0]
width  = I.shape[1]
channel = I.shape[2]
# Opencv store H as [0, 180) --> [0, 360)
H = I[:, :, 0].astype(np.int32)* 2
# Opencv store S as [0, 255] --> [0, 1]
S = I[:, :, 1].astype(np.float32) / 255.0
P = [ [], [] ]
#P = [ [0,0,1,1],[0,1,0,1] ]

def energy_E(V):
    w1 = 1.0
    w2 = 1.0
    return w1 * energy_E1(V)# + w2 * energy_E2(V)

def energy_E1(V):
    H_P = H[ P[0], P[1] ]
    V_P = V[ P[0], P[1] ]
    H_V_P = V_P.copy()
    H_V_P[V_P < 0.5] = 120
    H_V_P[V_P >= 0.5] = 240
    #print(V_P, H_V_P)
    S_P = S[ P[0], P[1] ]

    d = np.deg2rad(util.deg_distance(H_P, H_V_P))
    s = S_P
    
    e1 = np.multiply(d, s)
    e1 = np.sum(e1)
    return e1

def energy_E2(V):
    P_set, Q_set = util.PQ_N4(I, P)

    V_P = V[ P_set[0], P_set[1] ]    
    V_Q = V[ Q_set[0], Q_set[1] ]
    
    S_P = S[ P_set[0], P_set[1] ]
    S_Q = S[ Q_set[0], Q_set[1] ]
    H_P = H[ P_set[0], P_set[1] ]
    H_Q = H[ Q_set[0], Q_set[1] ]
    
    delta = util.delta( V_P, V_Q )
    s_max = np.max((S_P, S_Q), axis=0)
    d = np.deg2rad(util.deg_distance(H_P, H_Q))
    
    e2 = np.multiply( np.multiply( delta, s_max ), np.reciprocal(d) )
    #print(delta, s_max, np.multiply( delta, s_max ), np.reciprocal(d.astype(float)), e2)
    e2 = np.sum(e2)

    return e2

SEEDS = cv2.ximgproc.createSuperpixelSEEDS(HSV_image.shape[1], HSV_image.shape[0], HSV_image.shape[2], 400, 30)
SEEDS.iterate(HSV_image, 10)
'''
contour = SEEDS.getLabelContourMask()
for y in range(HSV_image.shape[0]):
    for x in range(HSV_image.shape[1]):
        if contour[y,x] == 255:
            color_image[y,x,0] = 255
            color_image[y,x,1] = 255
            color_image[y,x,2] = 255
cv2.imwrite("seeds.jpg", color_image)
'''
V = np.zeros(H.shape).reshape(-1)
N = V.shape[0]

grid_num = SEEDS.getNumberOfSuperpixels()
labels = SEEDS.getLabels()
for i in range(grid_num):
    P = [ [], [] ]
    for y in range(height):
        for x in range(width):
            if labels[y,x] == i:
                P[0].append(y)
                P[1].append(x)
    V0 = np.zeros(H.shape).astype(int)
    V1 = np.ones(H.shape).astype(int)
    #print(energy_E2(V0))
    print(i, energy_E(V0), energy_E(V1))

