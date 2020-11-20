# test
# NOT WORKING YET - IGNORE!
#
import numpy as np
import cv2
from numpy.lib.twodim_base import mask_indices

RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLACK = (0, 0, 0)
ORANGE = (255, 165, 0)

#
# These are the color definitions we are looking for in the picture
#
colors ={
    # All these are in HLS color space!
    # This is color definition GREEN
    # This is color definition (RED)
    "HSL_RED":{
        "weaker"    : np.array([ 0, 100,  20]),
        "stronger"  : np.array([ 5, 255, 255])
    },
    # This is color definition (RED2)
    "HSL_RED2":{
        "weaker"    : np.array([175, 100,  20]),
        "stronger"  : np.array([180, 255, 255])
    },
    "HSL_GREEN": {
        "weaker"    : np.array([ 40, 100,  20]),
        "stronger"  : np.array([ 75, 255, 200])
    },
    # This is color definition BLUE
    "HSL_BLUE":{
        "weaker"    : np.array([ 90, 100,  20]),
        "stronger"  : np.array([125, 255, 200])
    },
    # This is color definition (ORANGE)
    "HSL_ORANGE":{
        "weaker"    : np.array([10, 100,  20]),
        "stronger"  : np.array([20, 255, 200])
    }
}

def has_color(hsl, colorname, color):

    print(f"Testing image for {colorname}")

    mask = cv2.inRange(hsl, color["weaker"], color["stronger"])
    pixels = cv2.countNonZero(mask)
    print("{} pixels are matching color {} {} to {} ".format(pixels, colorname, color["weaker"], color["stronger"]))

    mask = cv2.inRange(hsl, color["weaker"], color["stronger"])

    #cv2.imshow("Mask",mask)
    #cv2.waitKey(0)

    return pixels > 0


def print_matrix(header, m):
    print("--- printing matrix {} ----".format(header))
    for x in m:
        for y in x:
            print(y, end='|')
        print()
    print("--- printing done ----")

#
# image color space is HLS
#
def test_pic(image, colorname, color):

    # display HLS image (convert it to RGB first otherwise it won't look OK on RGB screen)
    cv2.imshow('test', cv2.cvtColor(image, cv2.COLOR_HLS2RGB))
    cv2.waitKey(0)

    hsl = image

    if has_color(hsl, colorname, color):
        # add logic if image has the first color
        print(f"Image HAS color {colorname} in it")

    # if has_color(hsl, "TEST-R", { "weaker" : np.array([0, 0,  0]), "stronger": np.array([255, 0, 0])}):
    #     # add logic if image has the first color
    #     print(f"Image HAS color 1 in it")
    # if has_color(hsl, "TEST-G", { "weaker" : np.array([0, 0,  0]), "stronger": np.array([0, 255, 0])}):
    #     # add logic if image has the first color
    #     print(f"Image HAS color 2 in it")
    # if has_color(hsl, "TEST-B", { "weaker" : np.array([0, 0,  0]), "stronger": np.array([0, 0, 255])}):
    #     # add logic if image has the first color
    #     print(f"Image HAS color 3 in it")


def make_hsl(height, width, r, g, b):
    img = make_rgb(height, width, r,g,b)
    return cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

def make_hsv(height, width, r,g,b):
    img = make_rgb(height, width, r,g,b)
    return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

def make_rgb(height, width, r,g,b):
    # create a new image and fill it with white
    #img = np.ones((height,width,3),np.uint8)*255
    # create a new image and fill it with black
    img = np.zeros((height, width, 3), np.uint8)
    # fill a rect with the given 
    img[height//4:height-(height//4), width//4:width-(width//4)] = (b,g,r)
    return img

def tx_rgb_to_hsl(rgb):
    HSL_convert = cv2.cvtColor(np.uint8([[rgb]]), cv2.COLOR_RGB2HLS)
    return HSL_convert

def tx_rgb_to_hsv(rgb):
    HSV_convert = cv2.cvtColor(np.uint8([[rgb]]), cv2.COLOR_RGB2HSV)
    return HSV_convert

if __name__ == "__main__":

    print("HSV RED: {}".format(tx_rgb_to_hsv([255,0,0])))
    print("HSV GREEN: {}".format(tx_rgb_to_hsv([0, 255, 0])))
    print("HSV BLUE: {}".format(tx_rgb_to_hsv([0, 0, 255])))

    # 1x1
    red_hsv = tx_rgb_to_hsv([255, 0, 0])
    green_hsv = tx_rgb_to_hsv([0, 255, 0])
    blue_hsv = tx_rgb_to_hsv([0, 0, 255])

    red_mask = cv2.inRange(red_hsv, np.array([0,0,0]), np.array([10,255,255]))
    print(red_hsv.shape)
    print(red_mask.shape)
    print(red_mask)

    green_mask = cv2.inRange(green_hsv, np.array([40,0,0]), np.array([80,255,255]))
    print(green_hsv.shape)
    print(green_mask.shape)
    print(green_mask)

    blue_mask = cv2.inRange(blue_hsv, np.array([100,0,0]), np.array([140,255,255]))
    print(blue_hsv.shape)
    print(blue_mask.shape)
    print(blue_mask)



    # Now with an 8x8 matrix
    red_hsv_8x8 = make_hsv(8, 8, 128, 0, 0)
    cv2.imwrite("red_hsv_8x8.jpg", red_hsv_8x8)
    red_mask_8x8 = cv2.inRange(red_hsv_8x8, np.array([0,0,0]), np.array([10,255,255]))
    cv2.imwrite("red_mask.jpg", red_mask_8x8)

    print_matrix("red_hsv", red_hsv_8x8)
    print_matrix("red_mask", red_mask_8x8)

    print(red_hsv_8x8.shape)
    print(red_mask_8x8.shape)
    print(red_mask_8x8)


    # Now with an 8x8 matrix
    green_hsv_8x8 = make_hsv(8, 8, 0, 128, 0)
    cv2.imwrite("green_hsv_8x8.jpg", green_hsv_8x8)
    green_mask_8x8 = cv2.inRange(green_hsv_8x8, np.array([50,0,0]), np.array([70,255,255]))
    cv2.imwrite("green_mask.jpg", green_mask_8x8)

    print_matrix("green_hsv", green_hsv_8x8)
    print_matrix("green_mask", green_mask_8x8)

    print(green_hsv_8x8.shape)
    print(green_mask_8x8.shape)
    print(green_mask_8x8)


    # Now with an 8x8 matrix
    blue_hsv_8x8 = make_hsv(8, 8, 0, 0, 128)
    cv2.imwrite("blue_hsv_8x8.jpg", blue_hsv_8x8)
    blue_hsv_8x8_mask = cv2.inRange(blue_hsv_8x8, np.array([50,0,0]), np.array([70, 255,255]))
    cv2.imwrite("blue_hsv_8x8_mask.jpg", blue_hsv_8x8_mask)

    print_matrix("blue_hsv", blue_hsv_8x8)
    print_matrix("blue_mask", blue_hsv_8x8_mask)

    print(blue_hsv_8x8.shape)
    print(blue_hsv_8x8_mask.shape)
    print(blue_hsv_8x8_mask)

    # test_pic(red_hsl, "HSL_RED", colors["HSL_RED"])
    # test_pic(red_hsl, "HSL_RED2", colors["HSL_RED2"])
    # test_pic(green, "HSL_GREEN", colors["HSL_GREEN"])
    # test_pic(blue, "HSL_BLUE", colors["HSL_BLUE"])
    # test_pic(orange, "HSL_ORANGE", colors["HSL_ORANGE"])

    cv2.destroyAllWindows()

