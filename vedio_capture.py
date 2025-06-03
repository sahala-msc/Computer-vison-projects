import cv2
cap=cv2.VideoCapture(0)
while True:
    key,img=cap.read()
    cv2.imshow("test_vedio",img)
    cv2.waitKey(1)

