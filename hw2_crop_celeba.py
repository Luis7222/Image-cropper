# Dr. Kim's comments
# Prepare the original images and output folder
# Crop the faces using a pretrained face detection model
# This is to collect only frontal face for training GAN better

import os
import cv2

size = 64
dataPath = './celeba_original_202599/'  # prepare this folder with the original images
savePath = './celeba_cropped_64/'
if not os.path.exists(savePath):
    os.makedirs(savePath)  # create an output folder
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
li = os.listdir(dataPath)
count = 0
i = 0
n = len(li)
for fn in li:
    if fn[-4:] == '.jpg':
        image = cv2.imread(os.path.join(dataPath, fn))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 5, 5)
        if len(faces) != 1:
            pass
        else:
            x, y, w, h = faces[0]
            image_crop = image[y: y + w, x: x + w, :]
            image_resize = cv2.resize(image_crop, (size, size))
            cv2.imwrite(os.path.join(savePath, fn), image_resize)
            count += 1
    else:
        pass
    i += 1
    if i % 1000 == 0:
        print("%.2f" % ((i/n)*100), "%")  # print progress

print("total: %d / %d" % (count, len(li))) # number of selected images out of 200K
