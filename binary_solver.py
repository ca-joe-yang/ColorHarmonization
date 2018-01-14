import cv2
import numpy as np
import sys
import util
from scipy.optimize import minimize, rosen, rosen_der

image_filename = sys.argv[1]
color_image = cv2.imread(image_filename, cv2.IMREAD_COLOR)

height = color_image.shape[0]
width  = color_image.shape[1]

HSV_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)

I = HSV_image[0:20, 0:20]
height = I.shape[0]
width  = I.shape[1]
# Opencv store H as [0, 180) --> [0, 360)
H = I[:, :, 0].astype(np.int32)* 2
H = H.reshape(-1)
# Opencv store S as [0, 255] --> [0, 1]
S = I[:, :, 1].astype(np.float32) / 255.0
S = S.reshape(-1)
P = [ [1,2],[3,4] ]

a = 0.3
b = 0.4
def f(x):
	return np.sum(np.multiply( x-a, x-b ))

def energy_E(V):
    w1 = 10
    w2 = 1.0
    return w1 * energy_E1(V) + w2 * energy_E2(V)

def energy_E1(V):
    H_P = H[ P[0] + np.multiply(P[1],width) ]
    V_P = V[ P[0] + np.multiply(P[1],width) ]
    H_V_P = V_P.copy()
    H_V_P[V_P < 0.5] = 120
    H_V_P[V_P >= 0.5] = 240
    S_P = S[ P[0] + np.multiply(P[1],width) ]

    d = np.deg2rad(util.deg_distance(H_P, H_V_P))
    s = S_P
    
    e1 = np.multiply(d, s)
    e1 = np.sum(e1)
    return e1

def energy_E2(V):
    P_set, Q_set = util.PQ_N4(I, P)

    V_P = V[ P_set[0] + np.multiply(P_set[1],width) ]    
    V_Q = V[ Q_set[0] + np.multiply(Q_set[1],width) ]
    
    S_P = S[ P_set[0] + np.multiply(P_set[1],width) ]
    S_Q = S[ Q_set[0] + np.multiply(Q_set[1],width) ]
    H_P = H[ P_set[0] + np.multiply(P_set[1],width) ]
    H_Q = H[ Q_set[0] + np.multiply(Q_set[1],width) ]
    
    delta = util.delta( V_P, V_Q )
    s_max = np.max((S_P, S_Q), axis=0)
    d = np.deg2rad(util.deg_distance(H_P, H_Q))
    
    e2 = np.multiply( np.multiply( delta, s_max ), np.reciprocal(d) )
    #print(delta, s_max, np.multiply( delta, s_max ), np.reciprocal(d.astype(float)), e2)
    e2 = np.sum(e2)

    return e2


V = np.zeros(H.shape).reshape(-1)
V += 0.5
bnds = []
for _ in range(V.shape[0]):
	bnds.append((0,1))
res = minimize(energy_E, V, method='SLSQP')

