import cv2
import random
import colorsys
import numpy as np
import tensorflow as tf
import pytesseract
from core.config import cfg
import re
#import serial

from itertools import combinations
import math
import json

# If you don't have tesseract executable in your PATH, include the following:
# pytesseract.pytesseract.tesseract_cmd = r'<full_path_to_your_tesseract_executable>'
# Example tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract'

#ser = serial.Serial('/dev/ttyUSB0', 2000000, timeout=None)
hijau = 0
merah = 0
result = 0

def load_freeze_layer(model='yolov4', tiny=False):
    if tiny:
        if model == 'yolov3':
            freeze_layouts = ['conv2d_9', 'conv2d_12']
        else:
            freeze_layouts = ['conv2d_17', 'conv2d_20']
    else:
        if model == 'yolov3':
            freeze_layouts = ['conv2d_58', 'conv2d_66', 'conv2d_74']
        else:
            freeze_layouts = ['conv2d_93', 'conv2d_101', 'conv2d_109']
    return freeze_layouts

def load_weights(model, weights_file, model_name='yolov4', is_tiny=False):
    if is_tiny:
        if model_name == 'yolov3':
            layer_size = 13
            output_pos = [9, 12]
        else:
            layer_size = 21
            output_pos = [17, 20]
    else:
        if model_name == 'yolov3':
            layer_size = 75
            output_pos = [58, 66, 74]
        else:
            layer_size = 110
            output_pos = [93, 101, 109]
    wf = open(weights_file, 'rb')
    major, minor, revision, seen, _ = np.fromfile(wf, dtype=np.int32, count=5)

    j = 0
    for i in range(layer_size):
        conv_layer_name = 'conv2d_%d' %i if i > 0 else 'conv2d'
        bn_layer_name = 'batch_normalization_%d' %j if j > 0 else 'batch_normalization'

        conv_layer = model.get_layer(conv_layer_name)
        filters = conv_layer.filters
        k_size = conv_layer.kernel_size[0]
        in_dim = conv_layer.input_shape[-1]

        if i not in output_pos:
            # darknet weights: [beta, gamma, mean, variance]
            bn_weights = np.fromfile(wf, dtype=np.float32, count=4 * filters)
            # tf weights: [gamma, beta, mean, variance]
            bn_weights = bn_weights.reshape((4, filters))[[1, 0, 2, 3]]
            bn_layer = model.get_layer(bn_layer_name)
            j += 1
        else:
            conv_bias = np.fromfile(wf, dtype=np.float32, count=filters)

        # darknet shape (out_dim, in_dim, height, width)
        conv_shape = (filters, in_dim, k_size, k_size)
        conv_weights = np.fromfile(wf, dtype=np.float32, count=np.product(conv_shape))
        # tf shape (height, width, in_dim, out_dim)
        conv_weights = conv_weights.reshape(conv_shape).transpose([2, 3, 1, 0])

        if i not in output_pos:
            conv_layer.set_weights([conv_weights])
            bn_layer.set_weights(bn_weights)
        else:
            conv_layer.set_weights([conv_weights, conv_bias])

    # assert len(wf.read()) == 0, 'failed to read all data'
    wf.close()


def read_class_names(class_file_name):
    names = {}
    with open(class_file_name, 'r') as data:
        for ID, name in enumerate(data):
            names[ID] = name.strip('\n')
    return names

def load_config(FLAGS):
    if FLAGS.tiny:
        STRIDES = np.array(cfg.YOLO.STRIDES_TINY)
        ANCHORS = get_anchors(cfg.YOLO.ANCHORS_TINY, FLAGS.tiny)
        XYSCALE = cfg.YOLO.XYSCALE_TINY if FLAGS.model == 'yolov4' else [1, 1]
    else:
        STRIDES = np.array(cfg.YOLO.STRIDES)
        if FLAGS.model == 'yolov4':
            ANCHORS = get_anchors(cfg.YOLO.ANCHORS, FLAGS.tiny)
        elif FLAGS.model == 'yolov3':
            ANCHORS = get_anchors(cfg.YOLO.ANCHORS_V3, FLAGS.tiny)
        XYSCALE = cfg.YOLO.XYSCALE if FLAGS.model == 'yolov4' else [1, 1, 1]
    NUM_CLASS = len(read_class_names(cfg.YOLO.CLASSES))

    return STRIDES, ANCHORS, NUM_CLASS, XYSCALE

def get_anchors(anchors_path, tiny=False):
    anchors = np.array(anchors_path)
    if tiny:
        return anchors.reshape(2, 3, 2)
    else:
        return anchors.reshape(3, 3, 2)

def image_preprocess(image, target_size, gt_boxes=None):
    ih, iw    = target_size
    h,  w, _  = image.shape

    scale = min(iw/w, ih/h)
    nw, nh  = int(scale * w), int(scale * h)
    image_resized = cv2.resize(image, (nw, nh))

    image_paded = np.full(shape=[ih, iw, 3], fill_value=128.0)
    dw, dh = (iw - nw) // 2, (ih-nh) // 2
    image_paded[dh:nh+dh, dw:nw+dw, :] = image_resized
    image_paded = image_paded / 255.

    if gt_boxes is None:
        return image_paded

    else:
        gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]] * scale + dw
        gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]] * scale + dh
        return image_paded, gt_boxes

# helper function to convert bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, xmax, ymax
def format_boxes(bboxes, image_height, image_width):
    for box in bboxes:
        ymin = int(box[0] * image_height)
        xmin = int(box[1] * image_width)
        ymax = int(box[2] * image_height)
        xmax = int(box[3] * image_width)
        box[0], box[1], box[2], box[3] = xmin, ymin, xmax, ymax
    return bboxes

# function to add to JSON. ALAMAT FILE DISESUAIKAN KE data.json PADA KOMPUTER
def write_json(data, filename='F:\gamantaray\safinah_vision-master\core\data.json'): 
    with open(filename,'w') as f: 
        json.dump(data, f, indent=4) 
    
    
def draw_bbox(image, bboxes, info = False, counted_classes = None, show_label=True, allowed_classes=list(read_class_names(cfg.YOLO.CLASSES).values()), read_plate = False):
    #Kode Waba
    centroid_dict = dict()
    #End Kode Waba
    classes = read_class_names(cfg.YOLO.CLASSES)
    num_classes = len(classes)
    image_h, image_w, _ = image.shape
    print(image.shape)
    hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

    random.seed(0)
    random.shuffle(colors)
    random.seed(None)

    out_boxes, out_scores, out_classes, num_boxes = bboxes
    # print(bboxes)

    # temp_json = '{"info": "[ Cx Cy Xmin Ymin Xmax Ymax ]",}'
    temp_json = '{"contributors": "waba", "info": "[ Cx Cy Xmin Ymin Xmax Ymax ]"}'
    z = json.loads(temp_json)

    # temp_merah = []
    # temp_hijau = []

    temp_merah = {}
    temp_hijau = {}
    temp_max_merah = None
    temp_max_hijau = None



    for i in range(num_boxes):
        if int(out_classes[i]) < 0 or int(out_classes[i]) > num_classes: continue
        coor = out_boxes[i]
        fontScale = 0.5
        score = out_scores[i]
        class_ind = int(out_classes[i])
        class_name = classes[class_ind]
        if class_name not in allowed_classes:
            continue
        else:

            bbox_thick = int(0.6 * (image_h + image_w) / 600)
            c1, c2 = (int(coor[0]), int(coor[1])), (int(coor[2]), int(coor[3]))

            #Kode Waba
            cX = int((coor[0] + coor[2]) / 2)
            cY = int((coor[1] + coor[3]) / 2)

            centroid_dict[i] = (cX, cY, coor[0], coor[1], coor[2], coor[3])
            if (class_ind == 0):

    ###cek ini kotak di video gak
                cv2.rectangle(image, c1, c2, (0, 255, 0), bbox_thick)
                bbox_mess = '%s: %.2f' % (class_name, score)
                t_size = cv2.getTextSize(bbox_mess, 0, fontScale, thickness=bbox_thick // 2)[0]
                c3 = (c1[0] + t_size[0], c1[1] - t_size[1] - 3)
                cv2.rectangle(image, c1, (int(np.float32(c3[0])), int(np.float32(c3[1]))), (0, 255, 0), -1) #filled

                cv2.putText(image, bbox_mess, (c1[0], int(np.float32(c1[1] - 2))), cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale, (0, 0, 0), bbox_thick // 2, lineType=cv2.LINE_AA)
            else:


    ###cek ini kotak di video gak
                cv2.rectangle(image, c1, c2, (255, 0, 0), bbox_thick)
                bbox_mess = '%s: %.2f' % (class_name, score)
                t_size = cv2.getTextSize(bbox_mess, 0, fontScale, thickness=bbox_thick // 2)[0]
                c3 = (c1[0] + t_size[0], c1[1] - t_size[1] - 3)
                cv2.rectangle(image, c1, (int(np.float32(c3[0])), int(np.float32(c3[1]))), (255, 0, 0), -1) #filled

                cv2.putText(image, bbox_mess, (c1[0], int(np.float32(c1[1] - 2))), cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale, (0, 0, 0), bbox_thick // 2, lineType=cv2.LINE_AA)
            #End Kode Waba
            

            if info:
                # print("Object found: {}, Confidence: {:.2f}, BBox Coords (xmin, ymin, xmax, ymax): {}, {}, {}, {} ".format(class_name, score, coor[0], coor[1], coor[2], coor[3]))
                # print("Center found: {} ".format( centroid_dict[i]))
                #Kode Waba
                if (class_ind == 0):
                    # temp_hijau.append(list(centroid_dict[i]))
                    temp_hijau[centroid_dict[i]]=centroid_dict[i][1]
                    # deteksi = {
                    #     "bola_hijau" + str(i) : str(list(centroid_dict[i]))
                    # }


###tidak pakai contours karena pakai yolo object detection
                    if (len(temp_hijau) > 0):
###mendapatkan koordinat dari bola hijau terdekat (luas maksimum)
                        temp_max_hijau = max(temp_hijau, key=temp_hijau.get)
                        deteksi = {
                            "bola_hijau" : str(max(temp_hijau, key=temp_hijau.get))
                        }   
                    else:
                        deteksi = {
                            "bola_hijau" : "Tidak terdeteksi"
                        }
                else:
                    # print(centroid_dict[i][4])
                    # temp = []
                    # temp.append(centroid_dict[i][4])
                    # temp_merah.append(list(centroid_dict[i]))
                    temp_merah[centroid_dict[i]]=centroid_dict[i][1]
                    # deteksi = {
                    #     "bola_merah" + str(i) : str(list(centroid_dict[i]))
                    # }


###tidak pakai contours karena pakai yolo object detection
                    if (len(temp_merah) > 0):
###mendapatkan koordinat dari bola merah terdekat (luas maksimum)
                        temp_max_merah = max(temp_merah, key=temp_merah.get)
                        deteksi = {
                            "bola_merah" : str(max(temp_merah, key=temp_merah.get))
                        }
                        # print(max(temp_merah, key=temp_merah.get)[0])
                    else:
                        deteksi = {
                            "bola_merah" : "Tidak terdeteksi"
                        }

                z.update(deteksi) 
                # print(temp_max_merah[0])
                #End Kode Waba 
                # if (len(temp_merah) != None and len(temp_hijau) != None):
                #     if (len(temp_merah) > 0 and len(temp_hijau) > 0):
                # cv2.line(image, (temp_max_merah[0], temp_max_merah[1]), (temp_max_hijau[0], temp_max_hijau[1]), (0, 255, 0), thickness=1)


            if counted_classes != None:
                # print(counted_classes)
                height_ratio = int(image_h / 25)
                offset = 15
                cv2.rectangle(image, (0,0), (200,50), (0, 0, 0), -1) #filled
                for key, value in counted_classes.items():
                    cv2.putText(image, "{}: {}".format(key, value), (5, offset),
                            cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 2)
                    offset += height_ratio
    #Kode Waba

    # print(temp_merah)
    # print(max(temp_merah, key=temp_merah.get))
    # print(max(temp_merah))
    #if (len(temp_merah) > 0 and len(temp_hijau) > 0):
        #cv2.line(image, (int(temp_max_merah[4]), int(temp_max_merah[1])), (int(image_w/2), int(image_h/2)), (255, 0, 0), thickness=3)
        #cv2.line(image, (int(temp_max_hijau[2]), int(temp_max_hijau[1])), (int(image_w/2), int(image_h/2)), (0, 255, 0), thickness=3)
        #global hijau
        #hijau = temp_max_hijau[2]
    #KODENEW    
    if (len(temp_merah) > 0): #and len(temp_hijau) > 0):

###buat garis dari tengah ke bola###

        cv2.line(image, (int(temp_max_merah[4]), int(temp_max_merah[1])), (int(image_w/2), int(image_h/2)), (255, 0, 0), thickness=3)
        #cv2.line(image, (int(temp_max_hijau[2]), int(temp_max_hijau[1])), (int(image_w/2), int(image_h/2)), (0, 255, 0), thickness=3)
    
        #if (temp_max_hijau[2] > temp_max_merah[4]):
        #    print(temp_max_hijau[2])
        #else:
        #    print(temp_max_merah[4])
        
        global merah
        
        
###merah adalah koordinat x bola merah ke tengah
        merah = temp_max_merah[4]
        #print(temp_max_hijau[2])
    else :
        #global merah
        merah = 2000
        #print(merah)

    if (len(temp_hijau) > 0):
        #cv2.line(image, (int(temp_max_merah[4]), int(temp_max_merah[1])), (int(image_w/2), int(image_h/2)), (255, 0, 0), thickness=3)
        
###buat garis dari tengah ke bola###
        
        cv2.line(image, (int(temp_max_hijau[2]), int(temp_max_hijau[1])), (int(image_w/2), int(image_h/2)), (0, 255, 0), thickness=3)
    
        #if (temp_max_hijau[2] > temp_max_merah[4]):
        #    print(temp_max_hijau[2])
        #else:
        #    print(temp_max_merah[4])
        #print(temp_max_merah[4])
        global hijau


###hijau adalah koordinat x bola merah ke tengah
        hijau = temp_max_hijau[2]
        #print(temp_max_hijau[2])
    else :
        #global hijau
        hijau = 1000 
        #print(hijau)

    #if (hijau < 320) :
        #hijau = 640

    #if (merah > 320) :
        #merah = 0

###edit yang ini buat posisi kapal cenderung kanan atau kiri
    result = int((hijau - merah)//2)
###ditambah merah untuk menjadikan koordinat sesuai koordinat dari kiri
    result = int(result + merah)

########################### parsingan pake range terus kirim huruf ########################
    
    if (result > 0 and result <= 150) :
    	#ser.write('a'.encode())
    	print('a')

    if (result > 150 and result <= 300) :
    	#ser.write('b'.encode())
    	print('b')

    if (result > 300 and result <= 340) :
    	#ser.write('c'.encode())
    	print('c')

    if (result > 340 and result <= 490) :
    	#ser.write('d'.encode())
    	print('d')

    if (result > 490 and result <= 640) :
    	#ser.write('e'.encode())
    	print('e')
        
    if (hijau <= 640 and merah == 2000) :
    	#ser.write('f'.encode())
    	print('f')

    if (hijau == 1000 and merah >= 0 and merah <= 640) :
    	#ser.write('g'.encode())
    	print('g')

    if (hijau == 1000 and merah == 2000) :
    	#ser.write('h'.encode())
    	print('h')

##############################################################################################

## parsingan langsung angka ##

    #KIRIM DATA
    #result = str(result)
    #result = result + 'b'
    #print(result)

##############################

###ALAMAT FILE DISESUAIKAN KE data.json PADA KOMPUTER
    with open('F:\gamantaray\safinah_vision-master\core\data.json') as json_file:
        data = json.load(json_file) 
        
        temp = data['detect']
        temp.clear()
        
        # appending data to emp_details 
        temp.append(z) 
        # print(temp)
        
    write_json(data) 
    # print(json.dumps(z)) 
 
    #End Kode Waba
    return image


def bbox_iou(bboxes1, bboxes2):
    """
    @param bboxes1: (a, b, ..., 4)
    @param bboxes2: (A, B, ..., 4)
        x:X is 1:n or n:n or n:1
    @return (max(a,A), max(b,B), ...)
    ex) (4,):(3,4) -> (3,)
        (2,1,4):(2,3,4) -> (2,3)
    """
    bboxes1_area = bboxes1[..., 2] * bboxes1[..., 3]
    bboxes2_area = bboxes2[..., 2] * bboxes2[..., 3]

    bboxes1_coor = tf.concat(
        [
            bboxes1[..., :2] - bboxes1[..., 2:] * 0.5,
            bboxes1[..., :2] + bboxes1[..., 2:] * 0.5,
        ],
        axis=-1,
    )
    bboxes2_coor = tf.concat(
        [
            bboxes2[..., :2] - bboxes2[..., 2:] * 0.5,
            bboxes2[..., :2] + bboxes2[..., 2:] * 0.5,
        ],
        axis=-1,
    )

    left_up = tf.maximum(bboxes1_coor[..., :2], bboxes2_coor[..., :2])
    right_down = tf.minimum(bboxes1_coor[..., 2:], bboxes2_coor[..., 2:])

    inter_section = tf.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]

    union_area = bboxes1_area + bboxes2_area - inter_area

    iou = tf.math.divide_no_nan(inter_area, union_area)

    return iou


def bbox_giou(bboxes1, bboxes2):
    """
    Generalized IoU
    @param bboxes1: (a, b, ..., 4)
    @param bboxes2: (A, B, ..., 4)
        x:X is 1:n or n:n or n:1
    @return (max(a,A), max(b,B), ...)
    ex) (4,):(3,4) -> (3,)
        (2,1,4):(2,3,4) -> (2,3)
    """
    bboxes1_area = bboxes1[..., 2] * bboxes1[..., 3]
    bboxes2_area = bboxes2[..., 2] * bboxes2[..., 3]

    bboxes1_coor = tf.concat(
        [
            bboxes1[..., :2] - bboxes1[..., 2:] * 0.5,
            bboxes1[..., :2] + bboxes1[..., 2:] * 0.5,
        ],
        axis=-1,
    )
    bboxes2_coor = tf.concat(
        [
            bboxes2[..., :2] - bboxes2[..., 2:] * 0.5,
            bboxes2[..., :2] + bboxes2[..., 2:] * 0.5,
        ],
        axis=-1,
    )

    left_up = tf.maximum(bboxes1_coor[..., :2], bboxes2_coor[..., :2])
    right_down = tf.minimum(bboxes1_coor[..., 2:], bboxes2_coor[..., 2:])

    inter_section = tf.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]

    union_area = bboxes1_area + bboxes2_area - inter_area

    iou = tf.math.divide_no_nan(inter_area, union_area)

    enclose_left_up = tf.minimum(bboxes1_coor[..., :2], bboxes2_coor[..., :2])
    enclose_right_down = tf.maximum(
        bboxes1_coor[..., 2:], bboxes2_coor[..., 2:]
    )

    enclose_section = enclose_right_down - enclose_left_up
    enclose_area = enclose_section[..., 0] * enclose_section[..., 1]

    giou = iou - tf.math.divide_no_nan(enclose_area - union_area, enclose_area)

    return giou


def bbox_ciou(bboxes1, bboxes2):
    """
    Complete IoU
    @param bboxes1: (a, b, ..., 4)
    @param bboxes2: (A, B, ..., 4)
        x:X is 1:n or n:n or n:1
    @return (max(a,A), max(b,B), ...)
    ex) (4,):(3,4) -> (3,)
        (2,1,4):(2,3,4) -> (2,3)
    """
    bboxes1_area = bboxes1[..., 2] * bboxes1[..., 3]
    bboxes2_area = bboxes2[..., 2] * bboxes2[..., 3]

    bboxes1_coor = tf.concat(
        [
            bboxes1[..., :2] - bboxes1[..., 2:] * 0.5,
            bboxes1[..., :2] + bboxes1[..., 2:] * 0.5,
        ],
        axis=-1,
    )
    bboxes2_coor = tf.concat(
        [
            bboxes2[..., :2] - bboxes2[..., 2:] * 0.5,
            bboxes2[..., :2] + bboxes2[..., 2:] * 0.5,
        ],
        axis=-1,
    )

    left_up = tf.maximum(bboxes1_coor[..., :2], bboxes2_coor[..., :2])
    right_down = tf.minimum(bboxes1_coor[..., 2:], bboxes2_coor[..., 2:])

    inter_section = tf.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]

    union_area = bboxes1_area + bboxes2_area - inter_area

    iou = tf.math.divide_no_nan(inter_area, union_area)

    enclose_left_up = tf.minimum(bboxes1_coor[..., :2], bboxes2_coor[..., :2])
    enclose_right_down = tf.maximum(
        bboxes1_coor[..., 2:], bboxes2_coor[..., 2:]
    )

    enclose_section = enclose_right_down - enclose_left_up

    c_2 = enclose_section[..., 0] ** 2 + enclose_section[..., 1] ** 2

    center_diagonal = bboxes2[..., :2] - bboxes1[..., :2]

    rho_2 = center_diagonal[..., 0] ** 2 + center_diagonal[..., 1] ** 2

    diou = iou - tf.math.divide_no_nan(rho_2, c_2)

    v = (
        (
            tf.math.atan(
                tf.math.divide_no_nan(bboxes1[..., 2], bboxes1[..., 3])
            )
            - tf.math.atan(
                tf.math.divide_no_nan(bboxes2[..., 2], bboxes2[..., 3])
            )
        )
        * 2
        / np.pi
    ) ** 2

    alpha = tf.math.divide_no_nan(v, 1 - iou + v)

    ciou = diou - alpha * v

    return ciou

def nms(bboxes, iou_threshold, sigma=0.3, method='nms'):
    """
    :param bboxes: (xmin, ymin, xmax, ymax, score, class)

    Note: soft-nms, https://arxiv.org/pdf/1704.04503.pdf
          https://github.com/bharatsingh430/soft-nms
    """
    classes_in_img = list(set(bboxes[:, 5]))
    best_bboxes = []

    for cls in classes_in_img:
        cls_mask = (bboxes[:, 5] == cls)
        cls_bboxes = bboxes[cls_mask]

        while len(cls_bboxes) > 0:
            max_ind = np.argmax(cls_bboxes[:, 4])
            best_bbox = cls_bboxes[max_ind]
            best_bboxes.append(best_bbox)
            cls_bboxes = np.concatenate([cls_bboxes[: max_ind], cls_bboxes[max_ind + 1:]])
            iou = bbox_iou(best_bbox[np.newaxis, :4], cls_bboxes[:, :4])
            weight = np.ones((len(iou),), dtype=np.float32)

            assert method in ['nms', 'soft-nms']

            if method == 'nms':
                iou_mask = iou > iou_threshold
                weight[iou_mask] = 0.0

            if method == 'soft-nms':
                weight = np.exp(-(1.0 * iou ** 2 / sigma))

            cls_bboxes[:, 4] = cls_bboxes[:, 4] * weight
            score_mask = cls_bboxes[:, 4] > 0.
            cls_bboxes = cls_bboxes[score_mask]

    return best_bboxes

def freeze_all(model, frozen=True):
    model.trainable = not frozen
    if isinstance(model, tf.keras.Model):
        for l in model.layers:
            freeze_all(l, frozen)
def unfreeze_all(model, frozen=False):
    model.trainable = not frozen
    if isinstance(model, tf.keras.Model):
        for l in model.layers:
            unfreeze_all(l, frozen)

