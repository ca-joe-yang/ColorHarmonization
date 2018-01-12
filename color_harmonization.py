#! python3

import cv2
import sys
import numpy as np

import util
import importlib
importlib.reload(util)

HueTemplates = {
    "i"       : [( 0.00, 0.05)],
    "V"       : [( 0.00, 0.26)],
    "L"       : [( 0.00, 0.05), ( 0.25, 0.22)],
    "mirror_L": [( 0.00, 0.05), (-0.25, 0.22)],
    "I"       : [( 0.00, 0.05), ( 0.50, 0.05)],
    "T"       : [( 0.25, 0.50)],
    "Y"       : [( 0.00, 0.26), ( 0.50, 0.05)],
    "X"       : [( 0.00, 0.26), ( 0.50, 0.26)],
}
template_types = list(HueTemplates.keys())
M = len(template_types)
A = 180

deg_distance = util.deg_distance
H_cycle = util.H_cycle

class HueSector:

    def __init__(self, centor, width):
        # In Degree [0,180)
        self.centor = centor
        self.width  = width
        self.border = [(self.centor - self.width/2), (self.centor + self.width/2)]

    def is_in_sector(self, H):
        # True/False matrix if hue resides in the sector
        return deg_distance(H, self.centor) < self.width/2

    def distance_to_border(self, H):        
        H_1 = deg_distance(H, self.border[0])
        H_2 = deg_distance(H, self.border[1])
        H_distance = np.minimum(H_1, H_2)
        H_distance[self.is_in_sector(H)] = 0
        return H_distance

    def distance_to_centor(self, H):
        H_distance = deg_distance(H, self.centor)
        H_distance[self.is_in_sector(H)] = 0
        return H_distance

class HarmonicScheme:

    def __init__(self, m, alpha):
        self.m = m
        self.alpha = alpha
        self.sectors = []
        for t in HueTemplates[m]:
            centor = t[0] * H_cycle + alpha
            width  = t[1] * H_cycle
            sector = HueSector(centor, width)
            self.sectors.append( sector )

    def harmony_score(self, X):
        H = X[:, :, 0].astype(np.int32)
        S = X[:, :, 1].astype(np.int32)
        H_dis = self.hue_distance(H)
        H_dis = np.deg2rad(H_dis)
        #return np.sum(H_dis)
        return np.sum( np.multiply(H_dis, S) )

    def hue_distance(self, H):
        H_dis = [ sector.distance_to_border(H) for sector in self.sectors ]
        H_dis = np.asarray(H_dis)        
        H_dis = H_dis.min(axis=0)
        return H_dis

    def hue_shifted(self, H):
        H_d2b = [ sector.distance_to_border(H) for sector in self.sectors ]
        H_d2b = np.asarray(H_d2b)
        
        H_cls = np.argmin(H_d2b, axis=0)
        H_d2b = H_d2b.min(axis=0)

        H_ctr = np.zeros((H.shape))
        H_wid = np.zeros((H.shape))
        H_d2c = np.zeros((H.shape))
        for i in range(len(self.sectors)):
            s = self.sectors[i]
            mask = (H_cls == i)
            H_cen[H_sec == i] = s.c
            H_wid[H_sec == i] = s.w

            for y in range(H_sec.shape[0]):
                for x in range(H_sec.shape[1]):
                    #print(H_sec[y,x])
                    if H_sec[y,x] == i:
                        H_d2c[y,x] = s.centor_distance(H[y,x])
        H_sig = H_wid / 2

        H_gau = gaussian(H_d2c, 0, H_sig)
        #print(H_d2c)
        #print(H_sig)
        #print(H_gau.dtype)
        H_shi = H_cen + np.multiply(H_wid / 2, 1 - H_gau)
        H_shi = np.remainder(H_shi, H_cycle)
        return H_shi

def B(X):    
    F_matrix = np.zeros((M, A))
    for i in range(M):
        m = template_types[i]
        for j in range(A):
            print(i,j)
            alpha = 180/A * j
            harmomic_scheme = HarmonicScheme(m, alpha)
            F_matrix[i, j] = harmomic_scheme.harmony_score(X)

    print( np.argmin(F_matrix[5]) )
    print( F_matrix.min(axis=1) )
    best_m_idx = 5
    best_alpha = np.argmin(F_matrix[best_m_idx])
    (best_m_idx, best_alpha) = np.unravel_index( np.argmin(F_matrix), F_matrix.shape )
    best_m = template_types[best_m_idx]
    return HarmonicScheme(best_m, best_alpha)
