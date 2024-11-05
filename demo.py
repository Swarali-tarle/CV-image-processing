import cv2

img = cv2.imread('bike.jpg',1)
cv2.imshow('image',img)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('frame', gray)

img_resize = cv2.resize(img, (234,234))
cv2.imshow("resize image",img_resize)

img_crop = img[0:300, 0:400]
cv2.imshow('crop_image',img_crop)
cv2.imwrite('crop_copy.jpg',img_crop)

th, res = cv2.threshold(img, 130, 255, cv2.THRESH_BINARY)
cv2.imshow("threshold_img",res)
#threshold_binary_inv
#_trunc
#_tozero
#_otsu

img = cv2.imread('pic3.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
contours, hierarchy = cv2.findContours(binary, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
print("length of contours {}".format(len(contours)))
print(contours)
image_copy = img.copy()
image_copy = cv2.drawContours(image_copy, contours, -1, (0,255,0), thickness=2, lineType=cv2.LINE_AA)
cv2.imshow('Grayscale image', gray)
cv2.imshow('Drawn contours', image_copy)
cv2.imshow('Binary img',binary)

img = cv2.imread('pic5.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
detector = cv2.SimpleBlobDetector_create()
keypoints = detector.detect(binary)
blob_image = cv2.drawKeypoints(img, keypoints, None, (0,0,255) , cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow('blob_detection',blob_image)

# cap = cv2.VideoCapture(0);
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640,480))

cv2.waitKey(0)
cv2.destroyAllWindows()