import re
password = input("Enter string to test: ")
if re.match(r'[A-Za-z0-9@#$%^&+=]{8}', password):
    print('match')
else:
    print('no match')
# from PIL import Image
# from PIL import ImageEnhance
# from imutils.object_detection import non_max_suppression
# import cv2

# image = Image.open('images/test2.jpg')
# sharper = ImageEnhance.Color(image)
# image = sharper.enhance(0)
# sharper = ImageEnhance.Brightness(image)
# image = sharper.enhance(1.5)
# sharper = ImageEnhance.Contrast(image)
# image = sharper.enhance(1.3)
# img_name = "{}_edited.png".format('test2.png')
# image.save(img_name)
# print("{} written!".format(img_name))
