import cv2
from PIL import ImageEnhance, Image
import csv
import time
import datetime
from imutils.object_detection import non_max_suppression
import numpy as np
import pytesseract
from sys import platform as _platform
import re
import subprocess

if _platform == "linux" or _platform == "linux2":
    # linux
    pytesseract.pytesseract.tesseract_cmd = (
        "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
    )
elif _platform == "darwin":
    # MAC OS X
    pytesseract.pytesseract.tesseract_cmd = (
        "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
    )
elif _platform == "win32":
    # Windows
    pytesseract.pytesseract.tesseract_cmd = (
        "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
    )
elif _platform == "win64":
    # Windows 64-bit
    pytesseract.pytesseract.tesseract_cmd = (
        "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
    )

padding_x, padding_y = 0.01, 0.25
prediction_regex = r"[A-Za-z0-9]{7}"
new_width, new_height = 640, 640


def add_entry_to_csv(image_text, image_filename):
    """This function is used to write to the csv file."""

    current_date_time = datetime.datetime.now()
    current_date = current_date_time.strftime("%x")
    current_time = current_date_time.strftime("%X")

    with open("ofaj_data.csv", "a") as csv_file:
        csv_file = csv.writer(csv_file, quoting=csv.QUOTE_ALL)
        csv_file.writerow([current_date, current_time, image_filename, image_text])
        print("Entry added to CSV.")


def validate_prediction(predicted_text):
    """This function is used to comapare the prediction with regex."""

    validated = re.match(prediction_regex, predicted_text)

    if not validated:
        print("Predicted text could not be validated.")

    return validated


def decode_predictions(scores, geometry):
    # grab the number of rows and columns from the scores volume, then
    # initialize our set of bounding box rectangles and corresponding
    # confidence scores
    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []

    # loop over the number of rows
    for y in range(0, numRows):
        # extract the scores (probabilities), followed by the
        # geometrical data used to derive potential bounding box
        # coordinates that surround text
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]

        # loop over the number of columns
        for x in range(0, numCols):
            # if our score does not have sufficient probability,
            # ignore it
            if scoresData[x] < 0.5:
                continue

            # compute the offset factor as our resulting feature
            # maps will be 4x smaller than the input image
            (offsetX, offsetY) = (x * 4.0, y * 4.0)

            # extract the rotation angle for the prediction and
            # then compute the sin and cosine
            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            # use the geometry volume to derive the width and height
            # of the bounding box
            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]

            # compute both the starting and ending (x, y)-coordinates
            # for the text prediction bounding box
            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)

            # add the bounding box coordinates and probability score
            # to our respective lists
            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])

    # return a tuple of the bounding boxes and associated confidences
    return (rects, confidences)


padding = 0.05


def predict_text(net, image):
    # load the input image and grab the image dimensions
    orig = image.copy()
    (origH, origW) = image.shape[:2]

    # set the new width and height and then determine the ratio in change
    # for both the width and height
    (newW, newH) = (new_width, new_height)
    rW = origW / float(newW)
    rH = origH / float(newH)

    # resize the image and grab the new image dimensions
    image = cv2.resize(image, (newW, newH))
    (H, W) = image.shape[:2]

    # define the two output layer names for the EAST detector model that
    # we are interested -- the first is the output probabilities and the
    # second can be used to derive the bounding box coordinates of text
    layerNames = ["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]

    

    # construct a blob from the image and then perform a forward pass of
    # the model to obtain the two output layer sets
    blob = cv2.dnn.blobFromImage(
        image, 1.0, (W, H), (123.68, 116.78, 103.94), swapRB=True, crop=False
    )
    net.setInput(blob)
    (scores, geometry) = net.forward(layerNames)

    # decode the predictions, then  apply non-maxima suppression to
    # suppress weak, overlapping bounding boxes
    (rects, confidences) = decode_predictions(scores, geometry)
    boxes = non_max_suppression(np.array(rects), probs=confidences)

    # initialize the list of results
    results = []

    # loop over the bounding boxes
    for (startX, startY, endX, endY) in boxes:
        # scale the bounding box coordinates based on the respective
        # ratios
        startX = int(startX * rW)
        startY = int(startY * rH)
        endX = int(endX * rW)
        endY = int(endY * rH)

        # in order to obtain a better OCR of the text we can potentially
        # apply a bit of padding surrounding the bounding box -- here we
        # are computing the deltas in both the x and y directions
        dX = int((endX - startX) * padding_x)
        dY = int((endY - startY) * padding_y)

        # apply padding to each side of the bounding box, respectively
        startX = max(0, startX - dX)
        startY = max(0, startY - dY)
        endX = min(origW, endX + (dX * 2))
        endY = min(origH, endY + (dY * 2))

        # extract the actual padded ROI
        roi = orig[startY:endY, startX:endX]

        # in order to apply Tesseract v4 to OCR text we must supply
        # (1) a language, (2) an OEM flag of 4, indicating that the we
        # wish to use the LSTM neural net model for OCR, and finally
        # (3) an OEM value, in this case, 7 which implies that we are
        # treating the ROI as a single line of text
        config = "-l eng --oem 1 --psm 7"
        text = pytesseract.image_to_string(roi, config=config)

        # add the bounding box coordinates and OCR'd text to the list
        # of results
        results.append(((startX, startY, endX, endY), text))

    # sort the results bounding box coordinates from top to bottom
    results = sorted(results, key=lambda r: r[0][1])

    # loop over the results
    if(len(results)>0):
        ((startX, startY, endX, endY), text) = results[0]
    # display the text OCR'd by Tesseract
    # 172.57.100.6 -Server IP address

    else:
        text = ""

    print("Predicted text: {}".format(text))
    return text


def enhance_image(image):
    """Enhance The Image"""
    value = 15

    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv_image)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value  # brighter

    s[s > lim] = 255
    s[s <= lim] += value  # colorful

    final_hsv = cv2.merge((h, s, v))

    return final_hsv


def enhance_image_PIL(image_file):
    """Enhance The Image with PIL"""
    image = Image.open(image_file)
    sharper = ImageEnhance.Color(image)
    image = sharper.enhance(0)
    sharper = ImageEnhance.Brightness(image)
    image = sharper.enhance(1.5)
    sharper = ImageEnhance.Contrast(image)
    image = sharper.enhance(1.3)

    img_name = "temp.png"
    image.save(img_name)

    return img_name


def main():
    # load the pre-trained EAST text detector
    print("[INFO] loading EAST text detector...")
    net = cv2.dnn.readNet("frozen_east_text_detection.pb")

    cam = cv2.VideoCapture(0)

    while True:
        ret, frame = cam.read()

        cv2.imshow("Camera", frame)

        if not ret:
            print("CANT START CAM, closing...")
            break

        k = cv2.waitKey(1)

        if k % 256 == 27:
            # ESC pressed
            print("Closing...")
            break

        elif k % 256 == 32:
            # SPACE pressed
            image = frame

            millis = int(round(time.time() * 1000))
            image_filename = "datasets/ofaj_{}.png".format(millis)
            cv2.imwrite(image_filename, image)
            print("{} written!".format(image_filename))

            enhanced_image = cv2.imread(enhance_image_PIL(image_filename))
            # enhanced_image = enhance_image(image)

            predicted_text = predict_text(net, enhanced_image)

            if validate_prediction(predicted_text):
                add_entry_to_csv(predicted_text, image_filename)

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
main()