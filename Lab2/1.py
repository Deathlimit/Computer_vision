import cv2
import numpy as np

video = cv2.VideoCapture(0)

while True:
    ok, img = video.read()

    if not ok:
        break

    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


    lower_red1 = np.array([0, 133, 133])
    upper_red1 = np.array([7, 255, 255])

    lower_red2 = np.array([175, 133, 133])
    upper_red2 = np.array([180, 255, 255])


    mask1 = cv2.inRange(hsv_img, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv_img, lower_red2, upper_red2)


    red_mask = mask1 + mask2
    result = cv2.bitwise_and(img, img, mask=red_mask)


    cv2.imshow('Original', img)
    cv2.imshow('HSV', hsv_img)
    cv2.imshow('Red Mask', red_mask)
    cv2.imshow('Red Objects', result)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()