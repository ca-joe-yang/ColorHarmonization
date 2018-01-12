import cv2
import numpy as np

import math
import util
import sys

import color_harmonization
import importlib
importlib.reload(color_harmonization)
importlib.reload(util)

drawing = False # true if mouse is pressed
mode = True # if True, draw rectangle. Press 'm' to toggle to curve

ix,iy = -1,-1
alpha = 0
moving = False

def move_arrow(event,x,y,flags,param):
	global ix,iy,alpha,moving

	if event == cv2.EVENT_LBUTTONDOWN:
		moving = True
		ix,iy = x,y

	elif event == cv2.EVENT_MOUSEMOVE:
		if moving == True:
			idir = (ix - 255, iy - 255)
			nextdir = (x - 255, y - 255)
			alpha -= util.angle_clockwise(idir, nextdir)

			ix,iy = x,y

	elif event == cv2.EVENT_LBUTTONUP:
		moving = False

def draw_arrow(img, alpha):
	center = (255,255)
	x,y = 255 + (int)(255.0 * math.cos(alpha * np.pi / 180)), 255 + (int)(255.0 * math.sin(alpha * np.pi / 180))
	cv2.arrowedLine(img, center, (x,y), (0,255,0), 5)

image_filename = sys.argv[1]
color_image = cv2.imread(image_filename, cv2.IMREAD_COLOR)

height = color_image.shape[0]
width  = color_image.shape[1]

HSV_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)

canvas = util.draw_hue_histogram(HSV_image)

best_harmomic_scheme = color_harmonization.B(HSV_image)

cv2.namedWindow('hue_source')
cv2.setMouseCallback('hue_source',move_arrow)

while(1):

	best_harmomic_scheme.alpha = alpha

	overlay = util.draw_harmonic_scheme(best_harmomic_scheme, canvas)
	cv2.addWeighted(overlay, 0.5, canvas, 1 - 0.5, 0, canvas)

	cv2.imshow('hue_source', overlay)
	if cv2.waitKey(20) & 0xFF == 27:
		break

cv2.destroyAllWindows()