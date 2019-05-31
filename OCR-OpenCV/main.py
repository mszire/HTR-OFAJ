import cv2
from PIL import ImageEnhance
from PIL import Image
import csv
import datetime 
from imutils.object_detection import non_max_suppression
import numpy as np
import pytesseract
import argparse
import cv2

pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe' # only for windows users xD

# This function is used to to write the csv file and  append the the values. 
def write_file(input_text):
    dt=datetime.datetime.now()
    datef=dt.strftime("%x")
    timef=dt.strftime("%X")
    text = (input_text)

    with open('ofaj_data.csv', 'a') as fd:
        fd = csv.writer(fd, quoting=csv.QUOTE_ALL) 
        fd.writerow([datef , timef, text])

# This function is used to comapre the file.

def check_entery(check):
    import re
    predicted_text =(check)
    if re.match(r'[A-Za-z0-9]{,7}', predicted_text):
        write_file(predicted_text)
        print('Updated CSV file')
    else:
        print('Take Photo Properly')


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

padding =0.05


def predict_the_thing(img_name):
    # load the input image and grab the image dimensions
    image = cv2.imread(img_name)
    orig = image.copy()
    (origH, origW) = image.shape[:2]

    # set the new width and height and then determine the ratio in change
    # for both the width and height
    (newW, newH) = (640, 640)
    rW = origW / float(newW)
    rH = origH / float(newH)

    # resize the image and grab the new image dimensions
    image = cv2.resize(image, (newW, newH))
    (H, W) = image.shape[:2]

    # define the two output layer names for the EAST detector model that
    # we are interested -- the first is the output probabilities and the
    # second can be used to derive the bounding box coordinates of text
    layerNames = [
        "feature_fusion/Conv_7/Sigmoid",
        "feature_fusion/concat_3"]

    # load the pre-trained EAST text detector
    print("[INFO] loading EAST text detector...")
    net = cv2.dnn.readNet('frozen_east_text_detection.pb')

    # construct a blob from the image and then perform a forward pass of
    # the model to obtain the two output layer sets
    blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
        (123.68, 116.78, 103.94), swapRB=True, crop=False)
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
        dX = int((endX - startX) *0.1)
        dY = int((endY - startY) *0.25)

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
        config = ("-l eng --oem 1 --psm 7")
        text = pytesseract.image_to_string(roi, config=config)

        # add the bounding box coordinates and OCR'd text to the list
        # of results
        results.append(((startX, startY, endX, endY), text))

    # sort the results bounding box coordinates from top to bottom
    results = sorted(results, key=lambda r:r[0][1])

    # loop over the results
    ((startX, startY, endX, endY), text) = results[0]
    # display the text OCR'd by Tesseract
    # 172.57.100.6 -Server IP address
    print("OCR TEXT")
    print("========")
    print("{}\n".format(text))
    check_entery(text)

    # strip out non-ASCII text so we can draw the text on the image
    # using OpenCV, then draw the text and a bounding box surrounding
    # the text region of the input image
    text = "".join([c if ord(c) < 128 else "" for c in text]).strip()
    output = orig.copy()
    cv2.rectangle(output, (startX, startY), (endX, endY),
        (0, 0, 255), 2)
    cv2.putText(output, text, (startX, startY - 20),
        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    # show the output image
    cv2.imshow("Text Detection", output)
    cv2.waitKey(0)


#ENhance The Image 
def enhance_image(name):
    image = Image.open(name)
    sharper = ImageEnhance.Color(image)
    image = sharper.enhance(0)
    sharper = ImageEnhance.Brightness(image)
    image = sharper.enhance(1.5)
    sharper = ImageEnhance.Contrast(image)
    image = sharper.enhance(1.3)
    img_name = "{}_edited.png".format(name)
    image.save(img_name)
    print("{} written!".format(img_name))


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
        img_name = "opencv_frame_{}.png".format(img_counter)
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        img_counter += 1
        
        enhance_image(img_name)
        predictedtext = predict_the_thing( "{}_edited.png".format(img_name))

        # csvinsertionfunction(text)
