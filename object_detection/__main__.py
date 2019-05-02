import tarfile
import urllib
import numpy as np
import os
import sys
import tensorflow as tf
import time
import json
import cv2
import pymongo
import psutil
import multiprocessing

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
# ## Object detection imports
# Here are the imports from the object detection module.

from utils import label_map_util
from utils import visualization_utils as vis_util

# for elapse time
init_start_time = time.time()


# MONGO Private Key: 69886c33-6cba-4ad0-bc2e-09bbcd3eb28f

client = pymongo.MongoClient('mongodb+srv://tobyd:root@fmp-o7ona.gcp.mongodb.net/test?retryWrites=true')




db = client['FMP-JSON']
videoMongoConnection = db['Videos']


# directory where videos are being stored
WORKING_DIR = 'test/'





for input_vid in os.listdir(WORKING_DIR):

        # Creates directory for output files
        directory = 'output/' + str(input_vid) + '/frames/'


        if not os.path.exists(directory):
            os.makedirs(directory)
            os.makedirs(directory+ "logs/")


        # Get video from input
        cap = cv2.VideoCapture(WORKING_DIR + input_vid)


        # initial variables for video
        video_frame_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        legacy_warning_detected = False
        csv_string_for_log = ""



        # # MODEL PREPERARTION ----

        # What model to download.
        MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
        MODEL_FILE = MODEL_NAME + '.tar.gz'
        DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'


        # Path to frozen detection graph. This is the actual model that is used for the object detection.
        PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
        #
        # List of the strings that is used to add correct label for each box.
        PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')
        NUM_CLASSES = 90



        # # ## Download Model

        opener = urllib.request.URLopener()
        opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
        tar_file = tarfile.open(MODEL_FILE)
        for file in tar_file.getmembers():
          file_name = os.path.basename(file.name)
          if 'frozen_inference_graph.pb' in file_name:
            tar_file.extract(file, os.getcwd())



        # ## Load a (frozen) Tensorflow model into memory.
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')




        # ## Loading label map
        # Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine
        label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                                    use_display_name=True)
        category_index = label_map_util.create_category_index(categories)



        def get_objectname(index):
            return {
                1:"person",
                2:"bicycle",
                3:"car",
                6:"bus",
                8:"truck",
                18: "dog",
                19: "horse",
                20: "sheep",
                21: "cow"
            }[index]
        def display_load(prog):
            if (prog < 10):
                return "[**..................]"
            elif(prog < 20):
                return "[****................]"
            elif (prog < 30):
                return "[******..............]"
            elif (prog < 40):
                return "[********............]"
            elif (prog < 50):
                return "[**********..........]"
            elif (prog < 60):
                return "[************........]"
            elif (prog < 70):
                return "[**************......]"
            elif (prog < 80):
                return "[****************....]"
            elif (prog < 90):
                return "[******************..]"
            elif (prog < 100):
                return "[********************]"
        def bytes_to_gb(m):
            return round(((m / 1024) / 1024) / 1024, 2)


        # Size, in inches, of the output images.
        IMAGE_SIZE = (12, 8)


        # Create/Open both the JSON file object and CSV file object
        JSON_FILE = open('output/' + str(input_vid) + '/video.json', 'a')
        CSV_FILE = open('output/' + str(input_vid) + '/warning_frames.csv', 'a')




        frame_array = []
        video_json = {
            "video":input_vid,
            "frame_count":"",
            'frames':[None]
        }


        # frame array init



        #   START A TENSORFLOW SESSION
        with detection_graph.as_default():
            with tf.Session(graph=detection_graph) as sess:

                # setup video output stream
                # output_video = cv2.VideoWriter('output/' + str(input_vid) + "/output.avi",cv2.VideoWriter_fourcc(*"MJPG"), 30,(800,450))

                # whilst in the session, iterate through all frames
                for frame in range(1, video_frame_length):

                    # used to messure FPS
                    start_time = time.time()

                    # used to messure progression of detection
                    round_progress = round((frame/video_frame_length) * 100,2)


                    pool = multiprocessing.Pool(2)
                    # gets the image from input video

                    ret, image_np = cap.read()

                    # changes image size

                    image_np = cv2.resize(image_np, (800, 450))


                    object_array=[]

                    frame_json = ''
                    frame_pyton=''



                    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                    image_np_expanded = np.expand_dims(image_np, axis=0)
                    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')


                    # Each box represents a part of the image where a particular object was detected.
                    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')


                    # Each score represent how level of confidence for each of the objects.
                    # Score is shown on the result image, together with the class label.
                    scores = detection_graph.get_tensor_by_name('detection_scores:0')
                    classes = detection_graph.get_tensor_by_name('detection_classes:0')
                    num_detections = detection_graph.get_tensor_by_name('num_detections:0')


                    # Actual detection.
                    (boxes, scores, classes, num_detections) = sess.run(
                        [boxes, scores, classes, num_detections],
                        feed_dict={image_tensor: image_np_expanded})


                    # Visualization of the results of a detection.
                    vis_util.visualize_boxes_and_labels_on_image_array(
                        image_np,
                        np.squeeze(boxes),
                        np.squeeze(classes).astype(np.int32),
                        np.squeeze(scores),
                        category_index,
                        use_normalized_coordinates=True,
                        line_thickness=2)

                    # defaulted to false as nothing has been detected
                    warning_detected = False




                    memory_calculation = dict(psutil.virtual_memory()._asdict())

                   # declare a frame object
                    frame_pyton = {
                        "frame": frame,
                        "url": "output/"+str(input_vid)+"/frames/"+str(frame)+".png",
                        "object_array": [None],
                        "object_count": 0,
                        "warning_detected":0,
                        "proccess":[None]
                    }
                    proccess_python = {
                        "frame_rate":0,
                        "system_time": time.time(),
                        "cpu_usage_percent": 0,
                        "memory_usage_percent": memory_calculation['percent'],
                        "memory_free": memory_calculation['free'],
                        "memory_usage": memory_calculation['used']
                    }



                    # Change python object into JSON
                    frame_json = json.loads(json.dumps(frame_pyton))
                    proccess_json = json.loads(json.dumps(proccess_python))

                    # init declaration for later
                    object_json =""


                    obj_count =0
                    # iterate through the objects detected in the frame
                    for i, b in enumerate(boxes[0]):


                        #                  car                    bus                  truck               person                  dog                   horse                  sheep                   cow                   bycycle
                        if classes[0][i] == 3 or classes[0][i] == 6 or classes[0][i] == 8 or classes[0][i] == 1 or classes[0][i] == 18 or classes[0][i] == 19 or classes[0][i] == 20 or classes[0][i] == 21 or classes[0][i] == 2:


                            if scores[0][i] >= 0.5:



                                # get the center of the boxes found
                                mid_x = (boxes[0][i][1] + boxes[0][i][3]) / 2
                                mid_y = (boxes[0][i][0] + boxes[0][i][2]) / 2

                                apx_distance = round(((1 - (boxes[0][i][3] - boxes[0][i][1])) ** 4), 1)


                                # get the area of the box
                                box_area = round((((boxes[0][i][1] + boxes[0][i][3]) * 100) * (
                                        (boxes[0][i][0] + boxes[0][i][2]) * 100)) / 100,2)


                                # Fill out python object for object
                                object_python = {
                                    "object": None,
                                    "percentage_probability": round(scores[0][i] * 100,2),
                                    "box_points": str((boxes[0][i][0], boxes[0][i][1], boxes[0][i][2], boxes[0][i][3])),
                                    "box_area": box_area,
                                    "distance_away": apx_distance

                                }

                                # Convert Python object into JSON
                                object_json = json.loads(json.dumps(object_python))

                                # appends new object to array
                                object_array.append(object_json)


                                # Switches object name out for correct object


                                object_json['object'] = get_objectname(classes[0][i])
                                obj_count+=1

                                # puts a distance on the box on the frame
                                cv2.putText(image_np, '{}'.format(apx_distance), (int(mid_x * 800), int(mid_y * 450)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                                # puts a warning message on the frame
                                if apx_distance <= 0.6:
                                    if mid_x > 0.1 and mid_x < 0.7:
                                        cv2.putText(image_np, 'WARNING!!!', (200,200), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (50, 50, 255), 10)
                                       # changes warning to True
                                        warning_detected = True



                    #   Changes JSON warning depending on warning detected in frame
                    if warning_detected:
                        frame_json['warning_detected']= 1

                    else:
                        frame_json['warning_detected'] = 0

                    # gets the start and finish of every object detected for the CSV file
                    if warning_detected != legacy_warning_detected:
                        if warning_detected:
                            csv_string_for_log = "P,False," + str(frame) + ","
                            legacy_warning_detected = warning_detected
                        else:
                            csv_string_for_log += str(frame) + ",1,#6464ff,0"
                            CSV_FILE.write(csv_string_for_log + '\n')
                            legacy_warning_detected = warning_detected

                    # adds all objects to JSON array


                    # Saves the new Image with overlays
                    image_np = cv2.resize(image_np,(400,225))
                    cv2.imwrite(directory + str(frame) + '.png', image_np)
                    # output_video.write(image_np)


                    # for FPS calculation
                    end_time = time.time()


                    # FPS calculation
                    execution_time = round(1 / (end_time - start_time), 2)
                    proccess_json["frame_rate"]= execution_time
                    proccess_json["cpu_usage_percent"]= psutil.cpu_percent(interval=0.02)

                    frame_json["object_array"] = object_array
                    frame_json["object_count"] = obj_count
                    frame_json["proccess"] = proccess_json

                    frame_array.append(frame_json)
                    # only displays the progession every X times in this case its 5.
                    if frame % 10 == 0:
                        print(display_load(round_progress) + " @ " + str(
                            round(execution_time, 3)))

                video_json['frames'] = frame_array
                video_json['frame_count'] = video_frame_length


                JSON_FILE.close()

                x = videoMongoConnection.insert_one(video_json)
                print(x.inserted_id)
                # closes both files and releases memory
                CSV_FILE.close()
                cv2.destroyAllWindows()
