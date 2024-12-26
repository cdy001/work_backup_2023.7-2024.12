import cv2
import numpy as np

img_src = cv2.imread('23-11-14.jpg')
img_dst = img_src.copy()
# img_dst[620:900, 800:1500, :] = 0
# img_dst[2930:3140, 800:1500, :] = 0
# img_dst[2980:3150, 2150:2855, :] = 0
img_dst = img_dst[600:18080]
cv2.imwrite('1.jpg', img_dst)

# a = np.array([1, 3, 2, 5, 4, 7, 9, 8])
# b = np.array(['a', 'c', 'b', 'e', 'd', 'g', 'i', 'h'])


# a_b = sorted(zip(a, b), key=lambda x:x[0], reverse=True)
# print(a_b])

# idxs = np.argsort(a)
# print(idxs)

# img = cv2.imread('/data/cdy/adc/1.jpg')
# img_h = np.zeros_like(img[:10])
# img_w = np.zeros_like(img[:, :10])
# print(img.shape)
# print(np.vstack([img, img_h]).shape)
# print(np.hstack([img, img_w]).shape)