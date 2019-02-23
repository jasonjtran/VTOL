# import numpy as np
# import cv2 as cv
# import matplotlib.pyplot as plt
# img1 = cv.imread('images/mainimagerotate.jpg',0)          # queryImage
# img2 = cv.imread('images/template.jpg',0) # trainImage
# # Initiate ORB detector
# orb = cv.ORB_create()
# # find the keypoints and descriptors with ORB
# kp1, des1 = orb.detectAndCompute(img1,None)
# kp2, des2 = orb.detectAndCompute(img2,None)
# # create BFMatcher object
# bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
# # Match descriptors.
# matches = bf.match(des1,des2)
# # Sort them in the order of their distance.
# matches = sorted(matches, key = lambda x:x.distance)
# # Draw first 10 matches.
# outImg = np.empty((1,1))
# img3 = cv.drawMatches(img1,kp1,img2,kp2,matches[:10],outImg, flags=2)
# # plt.imshow(img3),plt.show()
# cv.imshow("show", img3)
# cv.waitKey(0)



import cv2
import numpy as np
from math import sqrt, acos


MATCH_COUNT = 10

full = cv2.imread("images/template.jpg")        
# crop = cv2.imread("images/mainimage.jpg") 
crop = cv2.imread("images/mainimagerotate.jpg") 
# crop = cv2.imread("images/mainimageright.jpg") 
img2 = cv2.cvtColor(full, cv2.COLOR_BGR2GRAY)
img1 = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

# Initiate ORB detector
orb = cv2.ORB_create(nfeatures=100000, scoreType=cv2.ORB_FAST_SCORE)

# find the keypoints and descriptors with ORB
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Match descriptors.
matches = bf.match(des1,des2)

# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)[:MATCH_COUNT]

if len(matches) >= MATCH_COUNT:
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in matches ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in matches ]).reshape(-1,1,2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    # matchesMask = mask.ravel().tolist()

    h,w = img1.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,M)
    mp = cv2.perspectiveTransform(np.float32([[w/2.0, h/2.0]]).reshape(-1,1,2), M)[0][0]
    
    cv2.circle(img2, (mp[0], mp[1]), 5, 255, -1)
    # img2 = cv2.polylines(img2,[np.int32(dst)],True,255, 5, cv2.LINE_AA)
else:
    print("Not enough matches! (minimum is %d matches)" % MATCH_COUNT)

# Draw matches.
img2 = cv2.polylines(img2, [np.int32(dst)], True, 255, 5, cv2.LINE_AA)
img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, flags=2)
result = cv2.polylines(full, [np.int32(dst)], True, (255, 255, 255), 5, cv2.LINE_AA)
# result = full.copy()
cv2.circle(result, (mp[0], mp[1]), 2, (255, 255, 255), -1)
cv2.circle(result, (mp[0], mp[1]), 10, (255, 255, 255), 2)

###########
# calculate info
###########

print("Center Coordinates: (%f, %f)" % (mp[0], mp[1]))

# vector of upper edge
vec = dst[3][0] - dst[0][0]
# print sqrt(np.dot(vec, vec))
# zoom factor crop width / full width
# zoom = img2.shape[1] / sqrt(np.dot(vec, vec))
# angle upper edge to x axis
angle = acos(np.dot(vec, np.array([1, 0])) / (sqrt(vec[0]**2 + vec[1]**2)))    

# print("zoom:", zoom)
print("Angle: %f" % angle)
print("Corners:")
print("\n".join([str(i[0]) for i in dst]))


# cv2.namedWindow("image", cv2.WINDOW_NORMAL)
cv2.namedWindow("image", cv2.WINDOW_AUTOSIZE)
cv2.imshow('image', img3)
cv2.waitKey(0)
cv2.imshow('image', result)
cv2.waitKey(0)