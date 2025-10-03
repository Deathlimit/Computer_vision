import cv2
import numpy as np

video = cv2.VideoCapture(0)

while True:
    ok, img = video.read()
    if not ok:
        break

    result_img = img

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

    if len(contours) > 0:
        for element in contours:
            if cv2.contourArea(element) > 500:

                moments = cv2.moments(element)

                center_x = int(moments['m10'] / moments['m00'])
                center_y = int(moments['m01'] / moments['m00'])

                x, y, w, h = cv2.boundingRect(element)

                cv2.rectangle(result_img, (x, y), (x + w, y + h), (0, 0, 0), 3)

                cv2.circle(result_img, (center_x, center_y), 5, (0, 0, 255), -1)

                info = [
                    f"Площадь: {cv2.contourArea(element):.0f}",
                    f"Центр: ({center_x}, {center_y})",
                    f"Размер прямоугольника: {w}x{h}"
                ]

                print(info)


    cv2.imshow('Black Rectangle', result_img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()