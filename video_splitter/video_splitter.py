import cv2
import os

IN_DIR = '/Users/tobydustin/PycharmProjects/tensorflow_cpu/video_splitter/videos/'
OUT_DIR = '/Users/tobydustin/PycharmProjects/tensorflow_cpu/video_splitter/frames/'
videos = os.listdir(IN_DIR)
print(videos)
for video in videos:
    out_path = OUT_DIR+video+'/frame/'
    os.makedirs(out_path)
    cap = cv2.VideoCapture(IN_DIR+video)
    for f in range(1,int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
        ret, frame = cap.read()
        frame = cv2.resize(frame, (800, 450))

        cv2.imwrite(out_path + str(f) + '.png',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
