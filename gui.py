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
moving = False
button_count = 8
image_count = 0

num_superpixels = 200

def mouse_event(event,x,y,flags,param):
    global ix,iy,alpha,moving,template_type
    if y <= util.canvas_h:

        if event == cv2.EVENT_LBUTTONDOWN:
            moving = True
            ix,iy = x,y

        elif event == cv2.EVENT_MOUSEMOVE:
            if moving == True:
                idir = (ix - 255, iy - 255)
                nextdir = (x - 255, y - 255)
                alpha -= util.angle_clockwise(idir, nextdir)
                if alpha > 360:
                    alpha -= 360
                elif alpha < 0:
                    alpha = 360 + alpha

                ix,iy = x,y

        elif event == cv2.EVENT_LBUTTONUP:
            moving = False
    else:

        if event == cv2.EVENT_LBUTTONDOWN:
            if y < util.canvas_h + util.button_h:
                if x < util.button_h:
                    template_type = list(color_harmonization.HueTemplates.keys())[0]

                elif x < util.button_h * 2:
                    template_type = list(color_harmonization.HueTemplates.keys())[1]

                elif x < util.button_h * 3:
                    template_type = list(color_harmonization.HueTemplates.keys())[2]

                elif x < util.button_h * 4:
                    template_type = list(color_harmonization.HueTemplates.keys())[3]

            else:
                if x < util.button_h:
                    template_type = list(color_harmonization.HueTemplates.keys())[4]

                elif x < util.button_h * 2:
                    template_type = list(color_harmonization.HueTemplates.keys())[5]

                elif x < util.button_h * 3:
                    template_type = list(color_harmonization.HueTemplates.keys())[6]

                elif x < util.button_h * 4:
                    template_type = list(color_harmonization.HueTemplates.keys())[7]


# Read image
image_filename = sys.argv[1]
color_image = cv2.imread(image_filename, cv2.IMREAD_COLOR)

height = color_image.shape[0]
width  = color_image.shape[1]

# YCrCb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2YCrCb)
# Y, Cr, Cb = cv2.split(YCrCb_image)
# Y = cv2.equalizeHist(Y)
# YCrCb_image = cv2.merge([Y, Cr, Cb])
# color_image = cv2.cvtColor(YCrCb_image, cv2.COLOR_YCrCb2BGR)

HSV_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)

# Count source hue histogram and draw
histo = util.count_hue_histogram(HSV_image)
canvas = util.draw_polar_histogram(histo)
ui_canvas = canvas.copy()

# calculate best harmomic scheme
best_harmomic_scheme = color_harmonization.B(HSV_image)
alpha = best_harmomic_scheme.alpha
template_type = best_harmomic_scheme.m

# draw best score template
overlay = util.draw_harmonic_scheme(best_harmomic_scheme, canvas)
cv2.addWeighted(overlay, 0.5, canvas, 1 - 0.5, 0, ui_canvas)
ui_canvas = util.draw_harmonic_scheme_border(best_harmomic_scheme, ui_canvas)

cv2.imshow('hue_source', ui_canvas.astype(np.uint8))

cv2.namedWindow('hue')
cv2.setMouseCallback('hue',mouse_event)

cv2.namedWindow('image')
print(alpha)
print(template_type)

while(1):

    best_harmomic_scheme.update_alpha(alpha)
    best_harmomic_scheme.update_template(template_type)

    # create new HSV image by best harmomic scheme using hue shifted
    new_HSV_image = best_harmomic_scheme.hue_shifted(HSV_image, num_superpixels)

    histo = util.count_hue_histogram(new_HSV_image)
    canvas = util.draw_polar_histogram(histo)

    overlay = util.draw_harmonic_scheme(best_harmomic_scheme, canvas)
    cv2.addWeighted(overlay, 0.5, canvas, 1 - 0.5, 0, ui_canvas)
    ui_canvas = util.draw_harmonic_scheme_border(best_harmomic_scheme, ui_canvas)
    button_canvas = util.draw_buttons(best_harmomic_scheme, color_harmonization.HueTemplates, ui_canvas)

    new_image = cv2.cvtColor(new_HSV_image, cv2.COLOR_HSV2BGR)

    cv2.imshow('hue', button_canvas.astype(np.uint8))
    cv2.imwrite(sys.argv[1] + 'hue.png', ui_canvas.astype(np.uint8))
        
    cv2.imshow('image', new_image.astype(np.uint8))

    k = cv2.waitKey(33)
    if k == 27:
        cv2.imwrite('new_image.png', new_image.astype(np.uint8))
        break
    elif k == ord('s'):
        cv2.imwrite(sys.argv[1] + '_new' + str(image_count) + '.png', new_image.astype(np.uint8))
        image_count += 1

cv2.destroyAllWindows()