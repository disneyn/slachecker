from numpy.lib.twodim_base import mask_indices
import neopixel
import board
import time
import cv2
import numpy as np

pixell = neopixel.NeoPixel(board.D18, 1, auto_write=False)
#
# If you have more diodes, initiate like:
# pixels = neopixel.NeoPixel(board.D18, NUM_DIODES, auto_write=False)
# and address with pixels[n]
#

#
# Standard RGB
#
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLACK = (0, 0, 0)
ORANGE = (255, 165, 0)

#
# These are the color definitions we are looking for in the picture
#
colors ={
    # This is color definition #1
    "PIC_COLOR_1": {
        "weaker" : np.array([80, 100, 100]),
        "stronger" : np.array([95,255,255])
    },
    # This is color definition #2
    "PIC_COLOR_2":{
        "weaker" : np.array([56, 180, 100]),
        "stronger" : np.array([65,255,255])
    },
    # This is color definition #3
    "PIC_COLOR_3":{
        "weaker" : np.array([20, 100, 100]),
        "stronger" : np.array([35,255,255])
    }
}



def has_color(hsv, colorname):
    c = colors[colorname]
    mask = cv2.inRange(hsv, c["weaker"], c["stronger"])
    pixels = cv2.countNonZero(mask)
    return pixels > 0


#
# main program
#
if __name__ == "__main__":

    path = "/home/pi/raspberry-pi-opencv/tests/arnika.jpg"

    while True:

        # Load updated picture.jpg
        img  = cv2.imread(path)
        imgCropped = img[180:935,0:400]
        hsv = cv2.cvtColor(imgCropped, cv2.COLOR_BGR2HSV)

        if has_color(imgCropped, "PIC_COLOR_1"):
            #
            # Found PIC_COLOR_1 in picture, setting LED to RED
            #
            print("Found DARK GREEN - setting diode color to GREEN")
            pixell.fill(GREEN)
            pixell.show()
        else:
            #
            # PIC_COLOR_1 not found in picture, keep checking
            #
            if has_color(img, "PIC_COLOR_2"):
                #
                # Found PIC_COLOR_2 in picture, setting LED to RED
                #
                print("Found GREEN - setting diode color to ORANGE")
                pixell.fill(ORANGE)
                pixell.show()
                print("SLA WILL EXPIRE TODAY")
            else:
                #
                # Neither PIC_COLOR_1 nor PIC_COLOR_2 found in picture, keep checking
                #
                if has_color(img, "PIC_COLOR_3"):
                    #
                    # Found PIC_COLOR_3 in picture, setting LED to GREEN
                    #
                    print("Found ORANGE - setting diode color to RED")
                    pixell.fill(RED)
                    pixell.show()
                    print("SLA WILL EXPIRE IN 1 HOUR")
                else:
                    #
                    # Found Neither PIC_COLOR_1, 2 nor 3 in picture, setting LED to BLACK
                    #
                    pixell.fill(BLACK)
                    pixell.show()
                    print("Found no PIC_COLOR_X - setting pixel to BLACK")
                    
        time.sleep(30)

        print("CHECKING AGAIN")
