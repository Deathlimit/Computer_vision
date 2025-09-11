import cv2
import time
IMAGE_FLAG = [0,3]

if (IMAGE_FLAG[0] == 1):
    flag1 = cv2.IMREAD_COLOR
    flag2 = cv2.IMREAD_REDUCED_GRAYSCALE_2
    flag3 = cv2.IMREAD_ANYDEPTH
    flag_hsv = cv2.COLOR_BGR2HSV

    cats1_img = cv2.imread(r'E:\\Projects\VisionKram\1\.venv\cats.png', flag1)
    cv2.namedWindow('Display window1', cv2.WINDOW_NORMAL)
    cv2.imshow('Display window1', cats1_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cats2_img = cv2.imread(r'E:\\Projects\VisionKram\1\.venv\cats.jpeg', flag2)
    cv2.namedWindow('Display window2', cv2.WINDOW_FULLSCREEN)
    cv2.imshow('Display window2', cats2_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cats3_img = cv2.imread(r'E:\\Projects\VisionKram\1\.venv\cats.jpg', flag3)
    cats4_img = cv2.imread(r'E:\\Projects\VisionKram\1\.venv\cats.jpg')
    cats_hsv = cv2.cvtColor(cats4_img, flag_hsv)
    cv2.namedWindow('Display window3', cv2.WINDOW_KEEPRATIO)
    cv2.namedWindow('Display window4', cv2.WINDOW_KEEPRATIO)
    cv2.imshow('Display window3', cats3_img)
    cv2.imshow('Display window4', cats_hsv)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

else:

    if (IMAGE_FLAG[1] == 0):
        cap = cv2.VideoCapture(r'E:\\Projects\VisionKram\1\.venv\cats.mp4', cv2.CAP_ANY)
        while(True):
            ret, frame = cap.read()
            if not (ret):
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            cv2.imshow('Frame', hsv)
            if cv2.waitKey(3) & 0xFF == 27:
                break

    elif (IMAGE_FLAG[1] == 1):
        video = cv2.VideoCapture(r'E:\\Projects\VisionKram\1\.venv\cat2.gif', cv2.CAP_ANY)
        ok, img = video.read()
        w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter("output.mp4", fourcc, 25, (w, h))
        while (True):
            ok, img = video.read()
            if not (ok):
                break
            cv2.imshow('img', img)
            video_writer.write(img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        video.release()
        cv2.destroyAllWindows()

    elif (IMAGE_FLAG[1] == 2):
        video = cv2.VideoCapture(0)
        print(video)
        w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter("ya.mp4", fourcc, 25, (w, h))
        while (True):
            ok, img = video.read()

            height = img.shape[0]
            width = img.shape[1]

            center_x = width // 2
            center_y = height // 2

            cv2.rectangle(img, (center_x - 25 // 2, center_y - 50), (center_x + 25 // 2, center_y + 50), (0, 0, 255), 5)
            cv2.rectangle(img, (center_x - 50, center_y - 25 // 2), (center_x + 50, center_y + 25 // 2), (0, 0, 255), 5)


            cv2.imshow('frame', img)
            video_writer.write(img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        video.release()
        cv2.destroyAllWindows()

    elif (IMAGE_FLAG[1] == 3):
        video = cv2.VideoCapture(0)
        print(video)
        w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter("ya.mp4", fourcc, 25, (w, h))
        while (True):
            ok, img = video.read()

            height = img.shape[0]
            width = img.shape[1]

            center_x = width // 2
            center_y = height // 2

            b, g, r = map(int, img[center_y, center_x])

            blue = abs(r - 0) + abs(g - 0) + abs(b - 255)
            green = abs(r - 0) + abs(g - 255) + abs(b - 0)
            red = abs(r - 255) + abs(g - 0) + abs(b - 0)



            if red <= green and red <= blue:
                color = (0, 0, 255)
            elif green <= red and green <= blue:
                color = (0, 255, 0)
            else:
                color = (255, 0, 0)

            cv2.rectangle(img, (center_x - 25 // 2, center_y - 50), (center_x + 25 // 2, center_y + 50), color, -1)
            cv2.rectangle(img, (center_x - 50, center_y - 25 // 2), (center_x + 50, center_y + 25 // 2), color, -1)

            cv2.imshow('frame', img)
            video_writer.write(img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        video.release()
        cv2.destroyAllWindows()



