пimport cv2
import numpy as np
import matplotlib.pyplot as plt
from google.colab.patches import cv2_imshow

fig, ax = plt.subplots()
buff = np.zeros((480,1))
img = cv2.imread(str(1)+".jpg")
image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
_,  img_bin = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
buf = np.array(img_bin)
buff = np.concatenate((buff, buf[:,0:260]), axis = 1)
for i in range(2, 9):
  img = cv2.imread(tr(i)+".jpg")
  image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
  _,  img_bin = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
  buf = np.array(img_bin)
  buff = np.concatenate((buff, buf[:,200:260]), axis = 1)
img = cv2.imread(str(10)+".jpg")
image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
_,  img_bin = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
buf = np.array(img_bin)
buff = np.concatenate((buff, buf[:,200:480]), axis = 1)
for j in range(0, 970, 10):
  for i in range(0, 480, 10):
    if (buff[i,j] < 200):
      ax.scatter(j, -i, c = 1)
    else:
      ax.scatter(j, -i, c ="#ffffff")
plt.show()
