#! python3

import cv2
import sys
import numpy as np

import color_harmonization
import util
import importlib
importlib.reload(color_harmonization)
importlib.reload(util)

image_filename = sys.argv[1]
color_image = cv2.imread(image_filename, cv2.IMREAD_COLOR)

height = color_image.shape[0]
width  = color_image.shape[1]

HSV_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)

canvas = util.draw_hue_histogram(HSV_image)

best_harmomic_scheme = color_harmonization.B(HSV_image)
print("Template Type  : ", best_harmomic_scheme.m)
print("Template Alpha : ", best_harmomic_scheme.alpha)

overlay = util.draw_harmonic_scheme(best_harmomic_scheme, canvas)
cv2.addWeighted(overlay, 0.5, canvas, 1 - 0.5, 0, canvas);
cv2.imwrite("hue_source.jpg", canvas)

'''
best_m = "T"
best_alpha = -35

best_template = HueTemplate(best_m, best_alpha)
new_H = best_template.hue_shifted(H)

HSV_image[:,:,0] = new_H

result_image = cv2.cvtColor(HSV_image, cv2.COLOR_HSV2BGR)

result_image = result_image.astype('int')
cv2.imwrite("result.jpg", result_image)
'''