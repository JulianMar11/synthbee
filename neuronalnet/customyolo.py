#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run a YOLO_v3 style detection model on test images.
"""

import colorsys
import os
import random
from timeit import default_timer as timer

import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from PIL import Image, ImageFont, ImageDraw
import cv2

from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
from yolo3.utils import letterbox_image

class YOLO(object):
    def __init__(self, path="logs/2018-07-0113_27_48.450243/ep011-loss23.257-val_loss23.492.h5", classes='input/classnames.txt', anchors='input/anchorstrain.txt', score=0.08):
        os.chdir("/Users/Julian/GitHub/synthbee/neuronalnet")
        
        #  self.model_path = 'model_data/yolov3-320.h5' # model path or trained weights path
        #  self.anchors_path =  "model_data/yolo_anchors.txt" #'model_data/yolo_anchors.txt'
        #  self.classes_path = 'model_data/coco_classes.txt'
        # self.model_path = "/media/bemootzer/SICHERUNG/Dokumente/BIENENDUMP/ServerPull/TD5_8000_512x288_manybees/ep003-loss50.155-val_loss49.848.h5"
        # self.model_path = "/media/bemootzer/SICHERUNG/Dokumente/BIENEN DUMP/ServerPull/Run 3.1 color/2018-06-26_02_30_19.628616ep018-loss39.721-val_loss39.770.h5"
        #self.model_path = "/media/bemootzer/SICHERUNG/Dokumente/BIENENDUMP/ServerPull/Run4/2018-06-2613_29_04.255107/ep015-loss31.279-val_loss31.855.h5"
        #### bestes model
        #self.model_path ="/media/bemootzer/SICHERUNG/Dokumente/BIENENDUMP/ServerPull/Run4/2018-06-2613_29_04.255107//ep010-loss31.293-val_loss31.551.h5"

        ### model mit viertel groesse
        #self.model_path = "/media/bemootzer/SICHERUNG/Dokumente/BIENENDUMP/ServerPull/Run5_viertel_groesse/2018-06-3015_40_45.782834/ep002-loss43.481-val_loss41.498.h5"

        self.model_path = path
        self.classes_path = classes
        self.anchors_path = anchors

        self.score = score
        self.iou = 0.5
        
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.model_image_size = (288, 512) # fixed size or (None, None), hw
        self.boxes, self.scores, self.classes = self.generate()

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        print("ANZAHL Klassen")
        print(num_classes)
        is_tiny_version = num_anchors <= 6 # default setting
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = tiny_yolo_body(Input(shape=(None,None,3)), num_anchors//2, num_classes) \
                if is_tiny_version else yolo_body(Input(shape=(None,None,3)), num_anchors//3, num_classes)
            self.yolo_model.load_weights(self.model_path) # make sure model, anchors and classes match
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                num_anchors/len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, and classes loaded.'.format(model_path))

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        random.seed(10101)  # Fixed seed for consistent colors across runs.
        random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        random.seed(None)  # Reset seed to default.

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2, ))
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                len(self.class_names), self.input_image_shape,
                score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes


    def validate_image(self, image):
        if self.model_image_size != (None, None):
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        # print(image_data.shape)
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })

        # print('Found {} boxes for {}'.format(len(out_boxes), 'img'))
        box_infos =  []

        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

          
            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            # print(label, (left, top), (right, bottom))

            # custom output
            box_infos.append([top, left, bottom, right, predicted_class, score])

        
        return image, box_infos



    def detect_image(self, image):
        start = timer()

        if self.model_image_size != (None, None):
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        # print(image_data.shape)
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })
        #print(out_classes)

        # print('Found {} boxes for {}'.format(len(out_boxes), 'img'))
        FONT_PATH = "/System/Library/Fonts/Keyboard.tff"
        font = ImageFont.truetype(font='/Library/Fonts/Arial.ttf', size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))  #'/home/bemootzer/.local/share/fonts/Caveat-Bold.ttf'
        #font = ImageFont.load("arial.pil")
        thickness = (image.size[0] + image.size[1]) // 300

        box_infos =  []

        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            # print(label, (left, top), (right, bottom))

            # custom output
            box_infos.append([top, left, bottom, right, predicted_class, score])

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            # My kingdom for a good redistributable image drawing library.
            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[c])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[c])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            del draw

        end = timer()
        # print(end - start)
        return image, box_infos

    def close_session(self):
        self.sess.close()


def track_video(yolo, video_path, conf_threshold, roi=False, relrect=(0,0,1,1), savevideo=False, polldetector=False):
    import cv2
    print(video_path)
    vid = cv2.VideoCapture(video_path)
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")

    cv2.namedWindow("result", cv2.WINDOW_NORMAL)

    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    prev_time = timer()
    
    predCenters = []

    background = []

    count_in = 0
    count_out = 0
    y_grenze = 0
    capopen = False

    if polldetector:
        cv2.namedWindow("bees", cv2.WINDOW_NORMAL)
        pollennet = YOLO(path="logs/2018-07-0313_31_51.276755/ep014-loss4.589-val_loss4.586.h5",classes='input/classnamespollen.txt',anchors='input/anchorstrain.txt',score=0.001) #score=0.005)

    while True:
        return_value, frame = vid.read()
        if roi:
            y1,x1,y2,x2 = relrect
            #print(relrect)
            #print(frame.shape)
            x1c = int(round(frame.shape[1]*x1,0))
            x2c = int(round(frame.shape[1]*x2,0))
            y1c = int(round(frame.shape[0]*y1,0))
            y2c = int(round(frame.shape[0]*y2,0))
            #print(x1c,x2c, y1c,y2c)

            #frame = frame[0:frame.shape[0]//2,0:frame.shape[1]//2,:]
            frame = frame[y1c:y2c,x1c:x2c:]

        if savevideo and not capopen:
            frame_height, frame_width, _ = frame.shape
            print(frame_width, frame_height)
            out = cv2.VideoWriter('/Users/Julian/Desktop/Ergebniss.avi',cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (frame_width, frame_height))
            capopen = True

        image = Image.fromarray(frame)
        image, predBoxes = yolo.detect_image(image)
        #print(predBoxes)

        result = np.asarray(image)

        if len(background) == 0:
            background = np.zeros(shape=result.shape, dtype=np.uint8)
            y_grenze = background.shape[0] / 2

            background = cv2.line(background, (0, int(y_grenze)), (background.shape[1], int(y_grenze)), color=[0,255,125], thickness=3)
     
        tmpCenters =  []
        tmpBees = []

        for box in predBoxes:
            # [top, left, bottom, right, predicted_class, score]
            if polldetector:
                padding = 20
                if box[0]-padding<0 or box[1]-padding<0 or box[2]+padding>frame.shape[0] or box[3]+padding>frame.shape[1]: #ignorepadding if box on edge
                    padding = 0
                beeroi = frame[box[0]-padding:box[2]+padding,box[1]-padding:box[3]+padding]

                roiimage = Image.fromarray(beeroi)

                roiimage, predBoxesbee = pollennet.detect_image(roiimage)
                resultroi = np.asarray(roiimage)
                cv2.imshow("bees", resultroi)

                for beebox in predBoxesbee:
                    cv2.rectangle(result,pt1=(box[1]+beebox[1]+padding,box[0]+beebox[0]+padding),pt2=(box[1]+beebox[3],box[0]+beebox[2]),color=(255,0,0),thickness=4)

            if box[5] < conf_threshold:
                continue
            center = [box[1] + (box[3]-box[1])/2, box[0] + (box[2]-box[0])/2, [random.randint(0,255),random.randint(0,255),random.randint(0,255)] ]
            tmpCenters.append(center)


        if len(tmpCenters) > 0:
            predCenters.append(tmpCenters)

        optimal_radius = [42.727, 85.4577, 128.1595, 171.023, 213.7345]
        if roi:
            optimal_radius = [42.727, 85.4577, 128.1595, 171.023, 213.7345] * 2

        # skip first frame since no previous center exist
        if len(predCenters) > 1:        
            for center in predCenters[-1]:
                min_dist = 100000000
                prev = None
   

                for t in [2,3,4,5,6] :
                    if len(predCenters) < t:
                        break
                    for prev_center in predCenters[-t]:
                        # Berechung der euklid. Distanz
                        dist = ((prev_center[0] - center[0])**2 + (prev_center[1] - center[1])**2 )**0.5
                        # Test, ob es sich um die kleinste Distanz in dem aktuellen Zeitschritt handelt
                        if dist < min_dist:
                            # falls ja, wird der aktuelle Vorgänger aus Vorgänger ausgesucht
                            min_dist = dist
                            prev = prev_center
                    #Test, ob das nächste Zentrum nah genug ist
                    if min_dist < optimal_radius[t-2]:
                        center[2] = prev[2]
                        background = cv2.line(background, (int(center[0]), int(center[1])), (int(prev[0]), int(prev[1])), center[2], 2)
                        
                        if t > 2:
                            print("Anzahl geskippter frames: " + str(t-2))

                        if prev[1] > y_grenze and center[1] < y_grenze:
                            count_out += 1
                        if prev[1] < y_grenze and center[1] > y_grenze:
                            count_in += 1
                        break
                    
            

        
        curr_time = timer()
        exec_time = curr_time - prev_time
        prev_time = curr_time
        accum_time = accum_time + exec_time
        curr_fps = curr_fps + 1
        if accum_time > 1:
            accum_time = accum_time - 1
            fps = "FPS: " + str(curr_fps) + "  in: " + str(count_in) + " out: " + str(count_out)
            curr_fps = 0
        cv2.putText(result, text=fps, org=(60, 120), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.0, color=(255, 0, 0), thickness=4)

        result = cv2.line(result, (0, int(y_grenze)), (result.shape[1], int(y_grenze)), color=[0,255,125], thickness=3)
        if savevideo:
            out.write(result)
        cv2.imshow("result", result)

        cv2.namedWindow("tracking", cv2.WINDOW_NORMAL)
        cv2.imshow("tracking", background)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    if savevideo:
        out.release()
    yolo.close_session()

def detect_video(yolo, video_path, conf_threshold,  quarterOfSize=False):
    import cv2
    vid = cv2.VideoCapture(video_path)
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")
    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    prev_time = timer()
    
   
    while True:
        return_value, frame = vid.read()
        if quarterOfSize:
            frame = frame[0:frame.shape[0]//2,0:frame.shape[1]//2,:]
        image = Image.fromarray(frame)
        image, predBoxes = yolo.detect_image(image)
        result = np.asarray(image)
        
        curr_time = timer()
        exec_time = curr_time - prev_time
        prev_time = curr_time
        accum_time = accum_time + exec_time
        curr_fps = curr_fps + 1
        if accum_time > 1:
            accum_time = accum_time - 1
            fps = "FPS: " + str(curr_fps)
            curr_fps = 0
        cv2.putText(result, text=fps, org=(25, 25), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5, color=(255, 0, 0), thickness=4)

        cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        cv2.imshow("result", result)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    yolo.close_session()


def detect_img(yolo):
    while True:
        img = input('Input image filename:')
        try:
            image = Image.open(img)
        except:
            print('Open Error! Try again!')
            continue
        else:
            r_image, _ = yolo.detect_image(image)
            r_image.show()
    yolo.close_session()



if __name__ == '__main__':
    detect_img(YOLO())
