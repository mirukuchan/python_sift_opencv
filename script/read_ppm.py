from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import codecs
import cv2
import math

local_address = './sample_img/sample_img3/Level3/' 
min_match_count = 10

### read ppm file
# img = Image.open(local_address+'1-003-1.ppm');
# im = np.array(img);

cv_img1 = cv2.imread(local_address+'3-001-1.ppm')
cv_img2 = cv2.imread(local_address+'3-001-2.ppm')

#bgr to rgb
cv_img1=cv2.cvtColor(cv_img1,cv2.COLOR_BGR2RGB)
cv_img2=cv2.cvtColor(cv_img2,cv2.COLOR_BGR2RGB)
# Initiate SIFT detector
sift = cv2.xfeatures2d.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(cv_img1, None)
kp2, des2 = sift.detectAndCompute(cv_img2, None)

flann_index_kdtree = 0

index_params = dict(algorithm = flann_index_kdtree, trees = 5)
search_params = dict(checks = 50) # 递归次数

flann = cv2.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(des1, des2, k=2) 
# store all the good matches
good = []
for m,n in matches:
    if m.distance < 0.7*n.distance:
        good.append(m)

if len(good) > min_match_count:
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()

    h = cv_img1.shape[0]
    w = cv_img1.shape[1]
    pts = np.float32([[0,0],[0,h-1],[w-1,h-1],[w-1,0]]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts, M)

    

    #cv_img2 = cv2.polylines(cv_img2,[np.int32(dst)],True,255,3,cv2.LINE_AA)

else:
    print("error")

draw_params = dict(matchColor = (0,255,0),
                    singlePointColor = None,
                    matchesMask = matchesMask,
                    flags = 2
)



cv_img3 = cv2.drawMatches(cv_img1, kp1, cv_img2, kp2, good, None, **draw_params)
cv_img3 = cv_img3.astype("uint8")

# b g r -> r g b
# b,g,r = cv2.split(cv_img3) 
# cv_img3 = cv2.merge([r,g,b])

# stitch
result = cv2.warpPerspective(cv_img2, np.linalg.inv(M),(cv_img1.shape[1] + cv_img2.shape[1], cv_img2.shape[0]))
result[0:cv_img1.shape[0], 0:cv_img1.shape[1]] = cv_img1

#calculte (translationx, translationy), rotation, (scalex, scaley), shear) from homography  matrix

def getComponents(normalised_homography):
    
  a = normalised_homography[0,0]
  b = normalised_homography[0,1]
  c = normalised_homography[0,2]
  d = normalised_homography[1,0]
  e = normalised_homography[1,1]
  f = normalised_homography[1,2]

  p = math.sqrt(a*a + b*b)
  r = (a*e - b*d)/(p)
  q = (a*d+b*e)/(a*e - b*d)

  translation = (c,f)
  scale = (p,r)
  shear = q
  theta = math.atan2(b,a)

  return (translation, theta, scale, shear)

# show pictures
plt.figure(1)
plt.subplot(2,2,1)
plt.imshow(cv_img1)
plt.subplot(2,2,2)
plt.imshow(cv_img2)
plt.subplot(2,2,3)
plt.imshow(cv_img3)
plt.subplot(2,2,4)
plt.imshow(result)
plt.show()

parameters = getComponents(M)

print(parameters)




# resize_img
# def resize_img(image, width_size, height_size):
#     row = np.shape(image)[0]
#     col = np.shape(image)[1]
#     # construct new image matrix
#     new_image = np.zeros((row * width_size, col * height_size, 3))
    
#     #print(new_image.shape)
#     # projection matrix
#     A = np.mat(([width_size, 0],[0, height_size]))
#     # reverse(A) * [new_x,new_y] = [x,y] 
#     for r in range(row * width_size):
#         for l in range(col * height_size):
#             v = np.dot(A.I, np.array([r,l]).T)
#             new_image[r,l] = image[int(v[0,0]),int(v[0,1])]
#     return new_image
# B = resize_img(im,2,2)

# print(im[0][2])
# print(B[0][5])
# plt.figure(1)
# plt.subplot(1,2,1)
# imgplot = plt.imshow(im)
# plt.subplot(1,2,2)
# uint8
# imgplot = plt.imshow(B.astype("uint8"))
# plt.show()

