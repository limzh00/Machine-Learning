import cv2
import matplotlib.pyplot as plt 

image = cv2.imread("custom_set/5.png", 0)
for i in range(len(image)):
    for j in range(len(image[0])):
        image[i][j] = 255 - image[i][j]
image = cv2.resize(image, (28,28))
retval, image = cv2.threshold(image, 100, 100, cv2.THRESH_TOZERO)
print(image.shape)
plt.imsave('custom_set/5_preprocess.png', image)