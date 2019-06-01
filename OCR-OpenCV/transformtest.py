import cv2
import numpy as np

image = cv2.imread('photoaf.jpg')

gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

# cv2.imshow("Text Detection", gray_image)
# cv2.waitKey(0)

value = 50

grey_new = np.where((255 - gray_image) < value,255,gray_image+value)

# cv2.imshow("Text Detection", grey_new)
# cv2.waitKey(0)


h, s, v = cv2.split(hsv_image)


cv2.imshow("Text Detection", h)
cv2.waitKey(0)
cv2.imshow("Text Detection", s)
cv2.waitKey(0)
cv2.imshow("Text Detection", v)
cv2.waitKey(0)


# lim = 255 - value
# v[v > lim] = 255
# v[v <= lim] += value

# final_hsv = cv2.merge((h, s, v))

