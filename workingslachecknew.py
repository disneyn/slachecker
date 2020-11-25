import neopixel
import board
import time
import cv2
import numpy as np

#
# If you have more diodes, initiate like:
# pixels = neopixel.NeoPixel(board.D18, NUM_DIODES, auto_write=False)
# and address with pixels[n]
#

#
# Standard RGB
#

pixell = neopixel.NeoPixel(board.D18, 3, auto_write=False)

RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLACK = (0, 0, 0)
ORANGE = (255, 165, 0)



path = "/home/pi/raspberry-pi-opencv/tests/picture.jpg"
img  = cv2.imread(path)
imgCropped = img[180:835,0:400]
hsv = cv2.cvtColor(imgCropped, cv2.COLOR_BGR2HSV)
weaker = np.array([80, 100, 100])
stronger = np.array([95,255,255])

mask = cv2.inRange(hsv, weaker, stronger)
pixels = cv2.countNonZero(mask)

if pixels > 100:
    print("Dark Green Color found - SLA TOMORROW")
    print("{} pixels are matching color {} to {} ".format(pixels, weaker, stronger))
    pixell[2] = (GREEN)
    pixell.show()
else :
    print("Not found colour")
    pixell[1] = (BLACK)
    pixell.show()
    

#check next color green
img  = cv2.imread(path)
imgCropped = img[180:835,0:400]
hsv = cv2.cvtColor(imgCropped, cv2.COLOR_BGR2HSV)
weaker = np.array([56, 180, 100])
stronger = np.array([65,255,255])

mask = cv2.inRange(hsv, weaker, stronger)
pixels = cv2.countNonZero(mask)

if pixels > 100:
    print("Green Color found - SLA TODAY")
    print("{} pixels are matching color {} to {} ".format(pixels, weaker, stronger))
    pixell[1] = (ORANGE)
    pixell.show()
else :
    print("Not found colour")
    pixell[1] = (BLACK)
    pixell.show()
    
#check next color orange
img  = cv2.imread(path)
imgCropped = img[180:835,0:400]
hsv = cv2.cvtColor(imgCropped, cv2.COLOR_BGR2HSV)
weaker = np.array([20, 100, 100])
stronger = np.array([35,255,255])

mask = cv2.inRange(hsv, weaker, stronger)
pixels = cv2.countNonZero(mask)

if pixels > 100:
    print("Orange Color found - SLA IN 1 HOUR")
    print("{} pixels are matching color {} to {} ".format(pixels, weaker, stronger))
    pixell[0] = (RED)
    pixell.show()
else :
    print("Not found colour")
    pixell[1] = (BLACK)
    pixell.show()
    
