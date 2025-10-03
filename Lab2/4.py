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


    kernel = np.ones((5, 5), np.uint8)

    better_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
    best_mask = cv2.morphologyEx(better_mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(best_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    result_img = img

    if len(contours) > 0:

        largest_contour = max(contours, key=cv2.contourArea)

        moments = cv2.moments(largest_contour)

        area = cv2.contourArea(largest_contour)


        m00 = moments['m00']
        m10 = moments['m10']
        m01 = moments['m01']


        cv2.drawContours(result_img, [largest_contour], -1, (0, 255, 0), 3)

        info = [
            f"Площадь: {area:.0f} пикселей",
            f"Момент m00: {m00:.0f}",
            f"Момент m10: {m10:.0f}",
            f"Момент m01: {m01:.0f}"
        ]

        print(info)


    cv2.imshow('Original', img)
    cv2.imshow('Red Mask', best_mask)
    cv2.imshow('Contours', result_img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()