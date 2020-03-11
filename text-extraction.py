import cv2
import numpy as np

# Load image, create mask, grayscale, Gaussian blur, Otsu's threshold
image = cv2.imread('test2_1.png')
original = image.copy()
blank = np.zeros(image.shape[:2], dtype=np.uint8)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
thresh = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY)[
    1]  # thresholding the image at a high intensity (since your text appears always to be white)

# Merge text into a single contour
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)  # do a closing operation to close the gaps

# Find contours
cnts = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]

# a = np.array([[1, 14], [2, 15], [3, 30]])
# b = np.array([[2, 12], [3, 12], [4, 14]])
# c = np.array([[2, 12], [3, 12], [4, 1123]])
#
# test = np.array((a, b , c))
# print(test)

new_img = image
boxList = []
for c in cnts:
    # Filter using contour area and aspect ratio
    x, y, w, h = cv2.boundingRect(c)
    area = cv2.contourArea(c)
    ar = w / float(h)
    if (area < 1000 and area > 10 and h < 200):  # filter areas -> what matches the patter of the text?
        # Find rotated bounding box
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        # fill the boxes - if color thinkness -1 its filled - if i.e. 2 your get a border
        cv2.drawContours(image, [box], 0, (255, 255, 255), -1)
        cv2.drawContours(blank, [box], 0, (255, 255, 255), -1)
        boxList.append(box)

# Bitwise operations to isolate text
extract = cv2.bitwise_and(thresh, blank)
extract = cv2.bitwise_and(original, original, mask=extract)

print(boxList)

# cv2.imshow('thresh', thresh)
# cv2.imshow('image', image)
# cv2.imshow('close', close)
# cv2.imshow('extract', extract)
cv2.imshow('final', new_img)
cv2.waitKey()
