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
margin=10
WHITE_LOWER   = np.array([0-margin, 0,   int( 1.0*255)])
WHITE_UPPER   = np.array([0+margin, 0,   int( 1.0*255)])
SILVER_LOWER  = np.array([0-margin, 0,   int(0.75*255)])
SILVER_UPPER  = np.array([0+margin, 0,   int(0.75*255)])
GRAY_LOWER    = np.array([0-margin, 0,   int(0.50*255)])
GRAY_UPPER    = np.array([0+margin, 0,   int(0.50*255)])
RED_LOWER     = np.array([0-margin, 255, int(0.50*255)])
RED_UPPER     = np.array([0+margin, 255 , int(0.50*255)])
MAROON_LOWER  = np.array([0-margin, 255, int(0.50*255)])
MAROON_UPPER  = np.array([0+margin, 255, int(0.50*255)])
LIME_LOWER    = np.array([int(120.0/360.0)-margin, 255,   int(1.0*255)])
LIME_UPPER    = np.array([int(120.0/360.0)+margin, 255,   int(1.0*255)])
NAVY_LOWER    = np.array([int(240.0/360.0)-margin, 255,   int(0.5*255)])
NAVY_UPPER    = np.array([int(240.0/360.0)+margin, 255,   int(0.5*255)])
FUCHSIA_LOWER = np.array([int(300.0/360.0)-margin, 255,   int(1.0*255)])
FUCHSIA_UPPER = np.array([int(300.0/360.0)+margin, 255,   int(1.0*255)])
PURPLE_LOWER  = np.array([int(300.0/360.0)-margin, 255,   int(0.5*255)])
PURPLE_UPPER  = np.array([int(300.0/360.0)+margin, 255,   int(0.5*255)])
CYAN_LOWER    = np.array([int(180.0/360.0)-margin, 255,   int(1.0*255)])
CYAN_UPPER    = np.array([int(180.0/360.0)+margin, 255,   int(1.0*255)])
TEAL_LOWER    = np.array([int(180.0/360.0)-margin, 255,   int(0.5*255)])
TEAL_UPPER    = np.array([int(180.0/360.0)+margin, 255,   int(0.5*255)])


GREEN_LOWER   = np.array([39, 255, 128])
GREEN_UPPER   = np.array([59, 255, 128])
BLUE_LOWER    = np.array([87,   0,   0])
BLUE_UPPER    = np.array([99, 255, 255])
YELLOW_LOWER  = np.array([25,   0,   0])
YELLOW_UPPER  = np.array([35, 255, 255])



colors ={
    # All these are in HLS color space!
    # This is color definition GREEN
    # This is color definition (RED)
    "HSV_RED":{
        "weaker"    : np.array([ 0,   0,   0]),
        "stronger"  : np.array([ 5, 255, 255])
    },
    # This is color definition (RED2)
    "HSV_RED2":{
        "weaker"    : np.array([175,   0,   0]),
        "stronger"  : np.array([180, 255, 255])
    },
    "HSV_GREEN": {
        "weaker"    : np.array([ 40,   0,   0]),
        "stronger"  : np.array([ 75, 255, 255])
    },
    # This is color definition BLUE
    "HSV_BLUE":{
        "weaker"    : np.array([ 90,   0,   0]),
        "stronger"  : np.array([125, 255, 255])
    },
    # This is color definition (ORANGE)
    "HSV_ORANGE":{
        "weaker"    : np.array([10,   0,   0]),
        "stronger"  : np.array([20, 255, 255])
    }
}


#
#
#
def countPixelsOfColor(hsv_img, colorRangeLower, colorRangeUpper):
    mask = cv2.inRange(hsv_img, colorRangeLower, colorRangeUpper)
    return cv2.countNonZero(mask)


def crop(img, x, y, w, h):
    return img[y:y + h, x:x + w]


def width(img):
    h, w, channels = img.shape
    return w


def height(img):
    h, w, channels = img.shape
    return h



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
    cv2.imshow('test', cv2.cvtColor(image, cv2.COLOR_HSV2RGB))
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


def make_hls(height, width, r, g, b):
    img = make_bgr(height, width, r, g, b)
    return cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

def make_hsv(height, width, r,g,b):
    img = make_bgr(height, width, r, g, b)
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


#
# Create an image in OpenCV's native BGR color space
# Background is black (0,0,0) with a filled rectangle
# in the middle.
# Rect has width = windowwidth /2 and height = windowheight/2
#
def make_bgr(height, width, r, g, b, white_bg = False):
    if white_bg:
        # create a new image and fill it with white
        img = np.ones((height,width,3),np.uint8)*255
    else:
        # create a new image and fill it with black
        img = np.zeros((height, width, 3), np.uint8)
    # fill a rect with the given width/2 and height/2
    # using slightly strange numpy syntax!
    img[height//4:height-(height//4), width//4:width-(width//4)] = (r, g, b)
    return img

#
#
#
def tx_rgb_to_hsl(rgb):
    HSL_convert = cv2.cvtColor(np.uint8([[rgb]]), cv2.COLOR_RGB2HLS)
    return HSL_convert

def tx_rgb_to_hsv(rgb):
    HSV_convert = cv2.cvtColor(np.uint8([[rgb]]), cv2.COLOR_RGB2HSV)
    return HSV_convert


def test_8x8():
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

    test_pic(red_hsv_8x8, "HSV_RED", colors["HSV_RED"])
    test_pic(red_hsv_8x8, "HSV_RED2", colors["HSV_RED2"])
    test_pic(green_hsv_8x8, "HSV_GREEN", colors["HSV_GREEN"])
    test_pic(blue_hsv_8x8, "HSV_BLUE", colors["HSV_BLUE"])
    #test_pic(orange_hsv_8x8, "HSL_ORANGE", colors["HSL_ORANGE"])

def test_simple(w, h):

    # RED
    x = make_bgr(w, h, 255, 0, 0)
    cv2.imwrite(f"{w}x{h}-rgb-red.jpg", x)
    hsv = cv2.cvtColor(x, cv2.COLOR_BGR2HSV)

    red_mask = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([10, 255, 255]))
    filtered_x = cv2.bitwise_and(x, x, mask=red_mask)
    cv2.imwrite(f"{w}x{h}-rgb-red-filtered.jpg", x)
    print(f"{cv2.countNonZero(red_mask)} in Red Mask")

    # GREEN
    x = make_bgr(w, h, 0, 255, 0)
    cv2.imwrite(f"{w}x{h}-rgb-green.jpg", x)
    hsv = cv2.cvtColor(x, cv2.COLOR_BGR2HSV)

    green_mask = cv2.inRange(hsv, np.array([40,0,0]), np.array([80,255,255]))
    print(f"{cv2.countNonZero(green_mask)} in Green Mask")

    filtered_x = cv2.bitwise_and(x, x, mask=green_mask)
    cv2.imwrite(f"{w}x{h}-rgb-green-filtered.jpg", x)

    # BLUE
    x = make_bgr(w, h, 0, 0, 255)
    cv2.imwrite(f"{w}x{h}-rgb-blue.jpg", x)
    hsv = cv2.cvtColor(x, cv2.COLOR_BGR2HSV)

    print_matrix("BLUE",  x)
    print_matrix("BLUE", hsv)
    blue_mask = cv2.inRange(hsv, np.array([1,1,1]), np.array([255,255,255]))
    print(f"Blue Mask has shape {blue_mask.shape} ")
    print_matrix("BLUE MASK", blue_mask)
    print(f"{cv2.countNonZero(blue_mask)} in Blue Mask")

    filtered_x = cv2.bitwise_and(x, x, mask=blue_mask)
    cv2.imwrite(f"{w}x{h}-rgb-blue-filtered.jpg", x)


def rgb2bgr(r, g, b):
    return b, g, r


def test_color():

    # Yellow

    # Now in BGR !
    img = make_bgr(320, 240, 255, 255, 0)
    cv2.imshow("color", img)
    cv2.waitKey(0)

    # Now convert to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    cv2.imshow("color", img)
    cv2.waitKey(0)

    # draw using BGR here! Always???
    cv2.rectangle(img, (10, 10), (110, 110), rgb2bgr(0, 0, 255), 1)

    cv2.imshow("color", img)
    cv2.waitKey(0)

    # ORANGE_MIN = np.array([5, 50, 50], np.uint8)
    # ORANGE_MAX = np.array([15, 255, 255], np.uint8)
    # BLUE_MIN = np.array([90, 0, 0], np.uint8)
    # BLUE_MAX = np.array([125, 255, 255], np.uint8)

    hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    frame_threshed = cv2.inRange(hsv_img, BLUE_LOWER, BLUE_UPPER)

    print(f"Number of nonzero = {cv2.countNonZero(frame_threshed)}")

    cv2.imshow("color", img)
    cv2.waitKey(0)

    cv2.imshow("color", frame_threshed)
    cv2.waitKey(0)


def test_ref():
    ref_blue = cv2.imread("ref_blue.jpg")
    ref_green = cv2.imread("ref_green.jpg")
    ref_yellow = cv2.imread("ref_yellow.jpg")

    hsv_blue = cv2.cvtColor(ref_blue, cv2.COLOR_BGR2HSV)
    hsv_green = cv2.cvtColor(ref_green, cv2.COLOR_BGR2HSV)
    hsv_yellow = cv2.cvtColor(ref_yellow, cv2.COLOR_BGR2HSV)


    # Test the blue image
    cv2.imshow("ref", ref_blue)
    cv2.waitKey(0)

    mask = cv2.inRange(hsv_blue, BLUE_LOWER, BLUE_UPPER)
    print(f"Number of nonzero = {cv2.countNonZero(mask)}")

    filtered = cv2.bitwise_and(ref_blue, ref_blue, mask=mask)
    cv2.imshow("ref", filtered)
    cv2.waitKey(0)


    # Test the green image
    cv2.imshow("ref", ref_green)
    cv2.waitKey(0)

    # Test the yellow image
    cv2.imshow("ref", ref_yellow)
    cv2.waitKey(0)



def test_bild():


    # Now in BGR !
    img = cv2.imread("testbild.png")
    cv2.imwrite("testbild-out.png", img)

    cv2.imshow("color", img)
    cv2.waitKey(0)

    # Now convert to RGB
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #cv2.imshow("color", img)
    #cv2.waitKey(0)

    # draw using BGR here! Always??? TEAL = #008080 or 0, 128, 128
    cv2.rectangle(img, (10, 10), (110, 110), rgb2bgr(0, 128, 128), 10)
    cv2.imshow("color", img)
    cv2.waitKey(0)

    # ORANGE_MIN = np.array([5, 50, 50], np.uint8)
    # ORANGE_MAX = np.array([15, 255, 255], np.uint8)
    # BLUE_MIN = np.array([90, 0, 0], np.uint8)
    # BLUE_MAX = np.array([125, 255, 255], np.uint8)

    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    print_matrix("hsv_testbild", hsv_img)

    cv2.imshow("color", hsv_img)
    cv2.waitKey(0)

    margin = 10
    GREEN_LOWER = np.array([int((120.0 / 360.0) * 100) - margin, 255, int(0.5 * 255)])
    GREEN_UPPER = np.array([int((120.0 / 360.0) * 100) + margin, 255, int(0.5 * 255)])
    TEAL_LOWER = np.array([int((180.0 / 360.0) * 100) - margin, 255, int(0.5 * 255)])
    TEAL_UPPER = np.array([int((180.0 / 360.0) * 100) + margin, 255, int(0.5 * 255)])

    print(TEAL_LOWER)
    print(TEAL_UPPER)

    h, w, channels = img.shape
    for x in range(0, h):
        for y in range(0, w):

            pixel = img[x, y]

            blue = pixel[0]
            green = pixel[1]
            red = pixel[2]

            if red == 0 and green == 128 and blue == 128:
                print("TEAL FOUND")

    #img = np.random.randint(255, size=(4, 4, 3))
    #blue, green, red = img[..., 0], img[..., 1], img[..., 2]
    #img[(green > 110) | ((blue < 100) & (red < 50) & (green > 80))] = [0, 0, 0]

    frame_threshed = cv2.inRange(hsv_img, TEAL_LOWER, TEAL_UPPER)

    print(f"Number of nonzero = {cv2.countNonZero(frame_threshed)}")

    cv2.imshow("color", img)
    cv2.waitKey(0)

    cv2.imshow("color", frame_threshed)
    cv2.waitKey(0)

if __name__ == "__main__":
    # test_8x8()

    # test_simple(16,16)
    # test_color()
    # test_bild()
    test_ref()

    cv2.destroyAllWindows()

