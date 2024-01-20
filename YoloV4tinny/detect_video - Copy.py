import os
import serial
# comment out below line to enable tensorflow outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
from core.functions import *
from tensorflow.python.saved_model import tag_constants
from PIL import Image
import cv2
import numpy as np
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
flags.DEFINE_string('weights', './checkpoints/yolov4-416',
                    'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')                   #size frame
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_string('video', './data/video/video.mp4', 'path to input video or set to 0 for webcam')
flags.DEFINE_string('output', None, 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_float('iou', 0.45, 'iou threshold')                        #BANYAKNYA KOTAK KOTAKNYA (Intersection of union)
flags.DEFINE_float('score', 0.70, 'score threshold')                    #BATAS bawah DETEKSI gambar 
flags.DEFINE_boolean('count', False, 'count objects within video')
flags.DEFINE_boolean('dont_show', False, 'dont show video output')
flags.DEFINE_boolean('info', False, 'print info on detections')
flags.DEFINE_boolean('crop', False, 'crop detections from images')
flags.DEFINE_boolean('plate', False, 'perform license plate recognition')

scalling = 1

batasLuasAtasKuning = 10000
batasLuasBawahKuning = 200

#setpoint stopping
LuasDermaga = 9000

port = '/dev/ttyUSB0'  
bautRate = 9600

#Atur jarak manuver
areaLurus = 20
areaBelokTipis = 40
areaBelokBesar = 40


#Rescale video
def rescale(frame, scale=0.75):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)
    return cv2.resize(frame, dimensions, interpolation=cv2.INTER_AREA)

def main(_argv):
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    input_size = FLAGS.size
    video_path = FLAGS.video
    # get video name by using split method
    video_name = video_path.split('/')[-1]
    video_name = video_name.split('.')[0]

    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])

        ##Koordinat area manuver
    batasLurus = {
        "+"     : areaLurus,
        "-"     : -areaLurus,
    }

    batasBelokTipis = {
        "+kanan"    : (areaLurus + areaBelokTipis),
        "+kiri"     : (areaLurus) -1,
        
        "-kanan"    : -(areaLurus) +1,
        "-kiri"     : -(areaLurus + areaBelokTipis),
    }

    batasBelokBesar = {
        "+kanan"    : (areaLurus + areaBelokTipis + areaBelokBesar),
        "+kiri"     : (areaLurus + areaBelokTipis) -1,

        "-kanan"    : -(areaLurus + areaBelokTipis) +1,
        "-kiri"     : -(areaLurus + areaBelokTipis + areaBelokBesar),
    }

    #Nyambungin ke arduino
    try:
        ser = serial.Serial(port, bautRate, timeout=None)
        arduino = 'ada'
        #startNyelem = 1000
    except:
        arduino = 'gada'
        print('------------gada arduino------------')

    if FLAGS.framework == 'tflite':
        interpreter = tf.lite.Interpreter(model_path=FLAGS.weights)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print(input_details)
        print(output_details)
    else:
        saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])
        infer = saved_model_loaded.signatures['serving_default']

    # begin video capture
    try:
        vid = cv2.VideoCapture(int(video_path))
    except:
        vid = cv2.VideoCapture(video_path)

    out = None

    if FLAGS.output:
        # by default VideoCapture returns float instead of int
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))

    frame_num = 0
    while True:
        return_value, frame = vid.read()
        if return_value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_num += 1
            image = Image.fromarray(frame)
        else:
            print('Video has ended or failed, try a different video format!')
            break
    
        frame_size = frame.shape[:2]
        image_data = cv2.resize(frame, (input_size, input_size))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        start_time = time.time()

        if FLAGS.framework == 'tflite':
            interpreter.set_tensor(input_details[0]['index'], image_data)
            interpreter.invoke()
            pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
            if FLAGS.model == 'yolov3' and FLAGS.tiny == True:
                boxes, pred_conf = filter_boxes(pred[1], pred[0], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
            else:
                boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
        else:
            batch_data = tf.constant(image_data)
            pred_bbox = infer(batch_data)
            for key, value in pred_bbox.items():
                boxes = value[:, :, 0:4]
                pred_conf = value[:, :, 4:]

        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=FLAGS.iou,
            score_threshold=FLAGS.score
        )

        if valid_detections == 0 :
            status, result = vid.read()
            result = rescale(result,scalling)
            global frametengah
            frametengah = result.shape[1]//2
            cv2.putText(result, "mode dermaga", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
            mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

            contours3, _ = cv2.findContours(mask_yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            global area3
            area3 = 0
            contoursList3 = []
            for i in contours3:
                area3 = cv2.contourArea(i)

                if area3 < batasLuasAtasKuning and area3 > batasLuasBawahKuning:
                    contoursList3.append(i)
                

            contours3 = tuple(contoursList3)

            global cxK
            global cyK

            ##Image segmenting DERMAGA
            if len(contours3) > 0:
                c = max(contours3, key=cv2.contourArea)
                
                area3 = cv2.contourArea(c)
                
                M = cv2.moments(c)
                if M['m00']!=0:
                    cxK = int(M['m10']/M['m00'])
                    cyK = int(M['m01']/M['m00'])
                    cv2.rectangle(result, (cxK-50,cyK-50), (cxK+50,cyK+50), (255,255,255), 1)
                cv2.putText(result, "Luas = "+str(area3), (cxK-53,cyK-53), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else :
                cxK = result.shape[1]//2
                cyK = result.shape[0]//2

            #manuver dermaga
            #-------------START----------#
            aKanan = cxK+batasLurus['+']
            aKiri = cxK+batasLurus['-']
            bKanan = cxK+batasBelokTipis['+kanan']
            bKiri = cxK+batasBelokTipis['-kiri']
            cKanan = cxK+batasBelokBesar['+kanan']
            cKiri = cxK+batasBelokBesar['-kiri']

        
            cv2.line(result, (0,result.shape[0]//2), (result.shape[1],result.shape[0]//2), (0,0,0), thickness=2)
            cv2.circle(result, (frametengah,result.shape[0]//2), 10, (0,0,255), thickness=2)
            
            cv2.line(result, (aKanan,0), (aKanan,result.shape[0]), (255,255,255), thickness=2)
            cv2.line(result, (aKiri,0), (aKiri,result.shape[0]), (255,255,255), thickness=2)
            cv2.line(result, (bKanan,0), (bKanan,result.shape[0]), (255,255,255), thickness=2)
            cv2.line(result, (bKiri,0), (bKiri,result.shape[0]), (255,255,255), thickness=2)
            cv2.line(result, (cKanan,0), (cKanan,result.shape[0]), (255,255,255), thickness=2)
            cv2.line(result, (cKiri,0), (cKiri,result.shape[0]), (255,255,255), thickness=2)

            if arduino == 'ada' :
                ##Lurus
                if frametengah > aKiri and frametengah < aKanan :
                    ser.write('a'.encode())
                    print('a')

                ##Belok kanan tipis
                elif frametengah > aKanan and frametengah < bKanan:
                    ser.write('b'.encode())
                    print('b')

                ##Belok kiri tipis
                elif frametengah > bKiri and frametengah < aKiri:
                    ser.write('c'.encode())
                    print('c')

                ##Belok kanan besar
                elif frametengah > bKanan and frametengah < cKanan:
                    ser.write('d'.encode())
                    print('d')

                ##Belok kiri besar
                elif frametengah > cKiri and frametengah < bKiri:
                    ser.write('e'.encode())
                    print('e')
  
                ##Belok kanan tajam
                elif frametengah > cKanan:
                    ser.write('f'.encode())
                    print('f')

                ##Belok kiri tajam
                elif frametengah < cKiri :
                    ser.write('g'.encode())
                    print('g')
                
                if area3 > LuasDermaga :
                    ser.write('h'.encode())
                    print('h')
            #-------------END------------#
        else :

            # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, xmax, ymax
            original_h, original_w, _ = frame.shape
            bboxes = utils.format_boxes(boxes.numpy()[0], original_h, original_w)

            pred_bbox = [bboxes, scores.numpy()[0], classes.numpy()[0], valid_detections.numpy()[0]]

            # read in all class names from config
            class_names = utils.read_class_names(cfg.YOLO.CLASSES)

            # by default allow all classes in .names file
            allowed_classes = list(class_names.values())
            
            # custom allowed classes (uncomment line below to allow detections for only people)
            # allowed_classes = ['person']

            # if crop flag is enabled, crop each detection and save it as new image
            if FLAGS.crop:
                crop_rate = 150 # capture images every so many frames (ex. crop photos every 150 frames)
                crop_path = os.path.join(os.getcwd(), 'detections', 'crop', video_name)
                try:
                    os.mkdir(crop_path)
                except FileExistsError:
                    pass
                if frame_num % crop_rate == 0:
                    final_path = os.path.join(crop_path, 'frame_' + str(frame_num))
                    try:
                        os.mkdir(final_path)
                    except FileExistsError:
                        pass          
                    crop_objects(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), pred_bbox, final_path, allowed_classes)
                else:
                    pass

            if FLAGS.count:
                # count objects found
                counted_classes = count_objects(pred_bbox, by_class = True, allowed_classes=allowed_classes)
                # loop through dict and print
                for key, value in counted_classes.items():
                    print("Number of {}s: {}".format(key, value))
                image = utils.draw_bbox(frame, pred_bbox, FLAGS.info, counted_classes, allowed_classes=allowed_classes, read_plate=FLAGS.plate)
            else:

                image = utils.draw_bbox(frame, pred_bbox, FLAGS.info, allowed_classes=allowed_classes, read_plate=FLAGS.plate)
        
            fps = 1.0 / (time.time() - start_time)
            print("FPS: %.2f" % fps)
            result = np.asarray(image)
            cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
        
            result = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        if not FLAGS.dont_show:
            cv2.imshow("result", result)
        
        if FLAGS.output:
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
