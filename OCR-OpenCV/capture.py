import cv2
from PIL import Image
import datetime
import time
cam = cv2.VideoCapture(0)

# cv2.namedWindow("Capture Image")

img_counter = 0
while True:
    ret, frame = cam.read()
    cv2.imshow("test", frame)
    
    if not ret:
        print("CANT START CAM, closing...")
        break

    k = cv2.waitKey(1)

    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        cam.release()
        cv2.destroyAllWindows()

        break
    elif k%256 == 32:
        # SPACE pressed
        # img_name = "opencv_frame_{}.png".format(img_counter)
        # cv2.imwrite(img_name, frame)
        # print("{} written!".format(img_name))
        # datt=datetime.datetime.now()
        # img_counter += 1
        # to_date= datt.strftime("%Y%m%d%H%M%S")
        # to_time= datt.strftime("%X")
        # print(datt)
        millis = int(round(time.time() * 1000))



        write_image = "datasets/opencv_frame_{}.png".format(millis)#'ofaj_{}_{}.png'.format(to_date,to_time)
        cv2.imwrite(write_image,frame)
        print('The orignal image is stored in the datasets folder ')
        