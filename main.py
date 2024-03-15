import cv2
from os import path


# Define pathing
basePath = path.dirname(path.relpath(__file__))
dataPath = path.join(basePath, 'data')
imagePath = path.join(dataPath, 'images')
aprilTagsPath = path.join(dataPath, 'april-tags')


image = cv2.imread(path.join(imagePath, 'image0000.jpg'))
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

cv2.imshow('', gray)
cv2.waitKey(0)