#! python3

import cv2
import numpy as np
import matplotlib.pyplot as plt

canvas_h = 600
canvas_w = 600
yc = int(canvas_h/2)
xc = int(canvas_w/2)
circle_r = 250
H_cycle = 180

def gaussian(X, m, S):
    T = X - m
    T_1 = np.multiply(T, T)
    T_2 = np.multiply(S, S)
    return np.exp( -T_1/(2*T_2) )

def deg_modulus(x):
    return np.remainder(x, H_cycle)

def deg_distance(H, a):
    H_a = np.abs(H - a)
    H_a = np.minimum(H_a, H_cycle-H_a)
    return H_a

def draw_hue_histogram(X):
    H = X[:, :, 0].astype(np.int32)
    S = X[:, :, 1].astype(np.int32)
    H_flat = H.flatten()
    
    count = np.zeros(H_cycle)
    for h in H_flat:
        count[ h ] += 1
    count /= np.max(count)
    count *= 250

    canvas = np.zeros((canvas_h, canvas_w, 3))
    cv2.circle(canvas, (yc, xc), circle_r, (255,255,255), -1)
    for i in range(H_cycle):
        theta = -i * (360/H_cycle) * np.pi / 180
        y1 = yc - int(circle_r * np.sin(theta))
        x1 = xc + int(circle_r * np.cos(theta))
        y2 = yc - int((circle_r-count[i]) * np.sin(theta))
        x2 = xc + int((circle_r-count[i]) * np.cos(theta))

        color_HSV = np.zeros((1,1,3), dtype=np.uint8)
        color_HSV[0,0,:] = [i,255,255]
        color_BGR = cv2.cvtColor(color_HSV, cv2.COLOR_HSV2BGR)
        B = int(color_BGR[0,0,0])
        G = int(color_BGR[0,0,1])
        R = int(color_BGR[0,0,2])
        cv2.line(canvas, (x1,y1), (x2,y2), (B,G,R), 3)
    cv2.circle(canvas, (yc, xc), circle_r+5, (255,255,255), 5)
    return canvas

   
def draw_harmonic_scheme(harmonic_scheme, canvas):
    overlay = canvas.copy()
    for sector in harmonic_scheme.sectors:
        center = sector.center
        width  = sector.width
        #print(center, width)
        start  = (center + width/2) * (360/H_cycle)
        end    = (center - width/2) * (360/H_cycle)
        #print(center, width, start, end)
        cv2.ellipse(overlay, (yc, xc), (circle_r,circle_r), 0, start, end, (0,0,0), -1)
    return overlay