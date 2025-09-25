import cv2
import numpy as np

PARAM = 0

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



    kernel = np.ones((5, 5), np.uint8)



    erosia= cv2.erode(red_mask, kernel, iterations=1)

    dilatacia = cv2.dilate(red_mask, kernel, iterations=1)

    opening = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)

    closing = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)


    result_original = cv2.bitwise_and(img, img, mask=red_mask)
    result_eroded = cv2.bitwise_and(img, img, mask=erosia)
    result_dilated = cv2.bitwise_and(img, img, mask=dilatacia)
    result_opening = cv2.bitwise_and(img, img, mask=opening)
    result_closing = cv2.bitwise_and(img, img, mask=closing)

    if (PARAM==0):
        cv2.imshow('Original Mask', red_mask)
        cv2.imshow('Erosia', erosia)
        cv2.imshow('Dilatacia', dilatacia)

        cv2.imshow('Opening (E+D)', opening)
        cv2.imshow('Closing (D+E)', closing)

    else:
        cv2.imshow('Original', img)
        cv2.imshow('Original+mask', result_original)
        cv2.imshow('Opening', result_opening)
        cv2.imshow('Closing', result_closing)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()