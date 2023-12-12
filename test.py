import matplotlib.pyplot as plt
import numpy as np
import cv2
import ISP_enhancement as ispe


# read image from data
img_path = "data\leno.jpg"
im = cv2.imread(img_path, cv2.IMREAD_COLOR)
resiz_ratio = 0.5
size= (round(np.shape(im)[0]*resiz_ratio), round(np.shape(im)[1]*resiz_ratio))
img = cv2.resize(im, size)
cv2.imshow("src", img)

# log transform
img_logt = ispe.log_transform(img, 100, 1)
cv2.imshow("img", img_logt)

# exponential transform
img_et = ispe.exp_tranform(img, 0.1, 1)
cv2.imshow("img", img_et)

# histogram equalization
img_he = ispe.hist_equalize(img)
cv2.imshow("HE", img_he)

# Gausssian filter
img_gauss = ispe.gaussfilt(img,0.5,3)
cv2.imshow("GAU", img_gauss)

# Laplacian filter

# kernel = np.array([ [0, 1, 0],
#                     [1,-4, 1],
#                     [0, 1, 0]]
#                 )
# img_lapfilt = cv2.filter2D(img, -1, kernel)
img_lapfilt = ispe.laplacaionfilt(img)
cv2.imshow("LapL", img_lapfilt)
cv2.waitKey(0)