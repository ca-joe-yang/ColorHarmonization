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
A = 360

deg_distance = util.deg_distance

class HueSector:

    def __init__(self, center, width):
        # In Degree [0,2 pi)
        self.center = center
        self.width  = width
        self.border = [(self.center - self.width/2), (self.center + self.width/2)]

    def is_in_sector(self, H):
        # True/False matrix if hue resides in the sector
        return deg_distance(H, self.center) < self.width/2

    def distance_to_border(self, H):        
        H_1 = deg_distance(H, self.border[0])
        H_2 = deg_distance(H, self.border[1])
        H_dist2bdr = np.minimum(H_1, H_2)
        return H_dist2bdr

    def debug_1(self, H, i):
        H_1 = deg_distance(H, self.border[0])
        H_2 = deg_distance(H, self.border[1])
        H_dist2bdr = np.minimum(H_1, H_2)
        

        canvas = util.draw_polar_histogram(H_1)
        cv2.imwrite(str(i)+"H1.jpg", canvas)

        canvas = util.draw_polar_histogram(H_2)
        cv2.imwrite(str(i)+"H2.jpg", canvas)

        canvas = util.draw_polar_histogram(H_dist2bdr)
        cv2.imwrite(str(i)+"H_dist2bdr_old.jpg", canvas)

        H_dist2bdr[self.is_in_sector(H)] = 0
        canvas = util.draw_polar_histogram(H_dist2bdr)
        cv2.imwrite(str(i)+"H_dist2bdr_new.jpg", canvas)
        
    def closest_border(self, H):
        H_1 = deg_distance(H, self.border[0])
        H_2 = deg_distance(H, self.border[1])
        H_cls_bdr = np.argmin((H_1, H_2), axis=0)
        H_cls_bdr = 2*(H_cls_bdr - 0.5)
        return H_cls_bdr


    def closest_border_dir(self, H):
        H_1 = deg_distance(H, self.border[0])
        H_2 = deg_distance(H, self.border[1])
        H_cls_bdr = np.argmin((H_1, H_2), axis=0)
        #H_cls_bdr = 2*(H_cls_bdr - 0.5)
        H_dir = np.zeros(H.shape)
        for i in range(2):
            mask = (H_cls_bdr == i)
            #print("util", util.deg_closest_direction(H, self.border[i]))
            #print("mask", mask)
            #print("rrrr", util.deg_closest_direction(H, self.border[i]) * mask)
            H_dir += util.deg_closest_direction(H, self.border[i]) * mask
        #print("H_dir", H_dir)
        return H_dir

    def distance_to_center(self, H):
        H_dist2ctr = deg_distance(H, self.center)
        return H_dist2ctr

class HarmonicScheme:

    def __init__(self, m, alpha):
        self.m = m
        self.alpha = alpha
        self.sectors = []
        for t in HueTemplates[m]:
            center = t[0] * 360 + alpha
            width  = t[1] * 360
            sector = HueSector(center, width)
            self.sectors.append( sector )

    def harmony_score(self, X):
        # Opencv store H as [0, 180) --> [0, 360)
        H = X[:, :, 0].astype(np.int32)* 2
        # Opencv store S as [0, 255] --> [0, 1]
        S = X[:, :, 1].astype(np.float32) / 255.0
        
        H_dis = self.hue_distance(H)
        H_dis = np.deg2rad(H_dis)
        #return np.sum(H_dis)
        return np.sum( np.multiply(H_dis, S) )

    def debug_1(self):
        H = np.asarray(range(360))
        for i in range(len(self.sectors)):
            sector = self.sectors[i]
            sector.debug_1(H, i)
            H_dist2bdr = sector.distance_to_border(H)
            H_dist2bdr[sector.is_in_sector(H)] = 0
            canvas = util.draw_polar_histogram(H_dist2bdr)
            #cv2.imwrite(str(i)+"y.jpg", canvas)

    def hue_distance(self, H):
        H_dis = []
        for i in range(len(self.sectors)):
            sector = self.sectors[i]
            H_dis.append(sector.distance_to_border(H))
            H_dis[i][sector.is_in_sector(H)] = 0
        H_dis = np.asarray(H_dis)        
        H_dis = H_dis.min(axis=0)
        return H_dis

    def debug_2(self):
        H = np.asarray(range(360))
        for i in range(len(self.sectors)):
            sector = self.sectors[i]
            #sector.debug_2(H, i)
            H_dist2ctr = sector.distance_to_center(H)
            width = sector.width
            center = sector.center
            H_gau = util.normalized_gaussian(H_dist2ctr, 0, width/1)
            for i in range(360):
                print(i, H_gau[i])
            H_dir = sector.closest_border(H)
            #H_dir = sector.closest_border_dir(H)
            
            canvas = util.draw_polar_histogram(H_dir)
            cv2.imwrite(str(i)+"_H_dir.jpg", canvas)
            
            H_tmp = np.multiply(width / 2, 1 - H_gau)
            for i in range(360):
                print(i, H_tmp[i])
            
            H_shf = -np.multiply( H_dir, H_tmp )
            canvas = util.draw_polar_histogram(H_shf)
            cv2.imwrite(str(i)+"_H_shf.jpg", canvas)
            for i in range(360):
                print(i, H_shf[i])
            #print("shift", H_shf)
            H_new = (center + H_shf)
            #print(H_new)
            H_new = np.remainder(H_new, 360)
            #print(H_new)

            H_diff = util.deg_distance(H, H_new)
            #print(H_diff.astype(int))
            for i in range(360):
                print(i, H_diff[i])
            canvas = util.draw_polar_histogram(H_diff)
            cv2.imwrite(str(i)+"_H_diff.jpg", canvas)
            
            H_new /= 2
            #print("new", H_new)
            H_new = H_new.astype(int)
            #print("new", H_new)
            canvas = util.draw_polar_histogram(H_new)
            cv2.imwrite(str(i)+"_H_new.jpg", canvas)



    def hue_shifted(self, X):
        Y = X.copy()
        H = X[:, :, 0].astype(np.int32)*2
        
        H_d2b = [ sector.distance_to_border(H) for sector in self.sectors ]
        H_d2b = np.asarray(H_d2b)
        
        H_cls = np.argmin(H_d2b, axis=0)
        H_d2b = H_d2b.min(axis=0)

        H_ctr = np.zeros((H.shape))
        H_wid = np.zeros((H.shape))
        H_d2c = np.zeros((H.shape))
        H_dir = np.zeros((H.shape))
        
        for i in range(len(self.sectors)):
            sector = self.sectors[i]
            mask = (H_cls == i)
            H_ctr[mask] = sector.center
            H_wid[mask] = sector.width
            H_dir += sector.closest_border(H) * mask
            #print(H_cls)
            #print(i,mask)
            #print(sector.center, sector.width)
            #print(sector.distance_to_center(H))
            #cv2.imwrite(str(i)+"tmp.jpg", sector.distance_to_center(H))
            #cv2.imwrite(str(i)+"mask.jpg", mask.astype(np.int32)*255)
            #print(mask)
            H_dist2ctr = sector.distance_to_center(H)
            H_dist2ctr[sector.is_in_sector(H)] = 0
            H_d2c += H_dist2ctr * mask
            #cv2.imwrite(str(i)+"d2c.jpg", H_d2c)
            
        #print(H_d2c)
        H_sgm = H_wid / 2
        H_gau = util.normalized_gaussian(H_d2c, 0, H_sgm)
        H_tmp = np.multiply(H_wid / 2, 1 - H_gau)
        H_shf = - np.multiply( H_dir, H_tmp )
        H_new = (H + H_shf).astype(np.int32)

        H_new = np.remainder(H_new, 360)
        H_new = (H_new/2).astype(np.uint8)
        #print(H_gau.max(), H_gau.min())
        #print(H_d2c.max())
        #print(H_sig)
        #print(H_gau.dtype)
        
        #print(H_gau)
        #cv2.imwrite("H_gau.jpg", H_gau*255)
        #print(H_tmp.max(), H_tmp.min())
        #print(H_ctr.max(), H_ctr.min())
        #cv2.imwrite("H_tmp.jpg", H_tmp)
        #H_shf[H_dir > 0] = np.multiply(H_wid/2, 1-H_gau)
        #H_shf = H_ctr - np.multiply( H_dir, np.multiply(H_wid / 2, 1 - H_gau) )
        #H_shf = H_ctr - np.multiply(H_wid / 2, 1 - H_gau)
        #print(H_shf.max(), H_shf.min())
        #H_shf = np.remainder(H_shf, H_cycle)
        #print(H_shf.max(), H_shf.min())
        #print(H_shf)
        Y[:,:,0] = H_new
        return Y

def B(X):
    F_matrix = np.zeros((M, A))
    for i in range(M):
        m = template_types[i]
        for j in range(A):
            print(i,j)
            alpha = 360/A * j
            harmomic_scheme = HarmonicScheme(m, alpha)
            F_matrix[i, j] = harmomic_scheme.harmony_score(X)

    #print( np.argmin(F_matrix[5]) )
    print( F_matrix.min(axis=1) )
    best_m_idx = 5
    best_alpha = np.argmin(F_matrix[best_m_idx])
    #(best_m_idx, best_alpha) = np.unravel_index( np.argmin(F_matrix), F_matrix.shape )
    best_m = template_types[best_m_idx]

    best_harmomic_scheme = HarmonicScheme(best_m, best_alpha)
    best_harmomic_scheme.debug_1()
    best_harmomic_scheme.debug_2()

    #best_harmomic_scheme.harmony_score(X, True)
    return best_harmomic_scheme