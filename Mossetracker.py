import numpy as np
import cv2

##Basierend auf http://www.bogotobogo.com/python/OpenCV_Python/python_opencv3_mean_shift_tracking_segmentation.php
##Klasse der Objekttracker
class trackobject():
    #Initiieren neuen Tracker auf deinem bestimmten Frame mit Objekt und einer Lebensdauer
    def __init__(self, count, object, frame):
        y = object['topleft']['y']
        h = object['bottomright']['y'] - object['topleft']['y']
        x = object['topleft']['x']
        w = object['bottomright']['x'] - object['topleft']['x']

        self.object = object
        self.trackhistory = []
        self.track_window = (x, y, w, h)
        self.IDobject = object['ID']
        self.yolo = object
        center = getcenter(self.track_window)
        self.distance = getdistance(center, frame)

        # set up the ROI for tracking
        self.roi = frame[y:y+h, x:x+w]
        self.hsv_roi = cv2.cvtColor(self.roi, cv2.COLOR_RGB2HSV)

        self.maskinv = cv2.inRange(self.hsv_roi, np.array((0, 0, 0)), np.array((180, 255, 160)))
        self.mask = cv2.bitwise_not(self.maskinv)

        self.roi_hist = cv2.calcHist([self.hsv_roi], [0], self.maskinv, [180], [0, 180])
        cv2.normalize(self.roi_hist, self.roi_hist, 0, 255, cv2.NORM_MINMAX)

        self.term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

        self.maxcount = count
        self.currentcount = 0
        self.trackhistory = []

        '''
        #Tests zum Anpassen der Maske
        for h in range(0, 180, 15):
            for v in range(0, 255, 50):
                for s in range(0, 255, 50):
                    lower = np.array([0, 0, 0])
                    higher = np.array([h, v, s])
                    mask = cv2.inRange(self.hsv_roi, lower, higher)
                    cv2.imwrite("NewObject_ID_" + str(self.IDobject) + str(h) + "V_" + str(v) + "S_" + str(s) + ".jpg",mask)
        
        for s in range(100, 255, 20):
                    lower = np.array([0, 0, 0])
                    higher = np.array([180, 255, s])
                    mask = cv2.inRange(self.hsv_roi, lower, higher)
                    cv2.imwrite("NewObject_ID_" + str(self.IDobject) + "Param_" + "S_" + str(s) + ".jpg",mask)

'''

        #Testzur Ausgabe der Maske

        cv2.imwrite("NewObject_ID_"+ str(self.IDobject) +"_maskinv.jpg", self.maskinv)
        cv2.imwrite("NewObject_ID_"+ str(self.IDobject) +"_maskfinal.jpg", self.mask)
        cv2.imwrite("NewObject_ORIGNAL_ID_" + str(self.IDobject) +".jpg", self.roi)

    #Methode zum Ersetzen des bestehenden Objektes durch ein neues Objekt
    #Es wird eine neue Region of Interest erzeugt
    def update(self, object, frame):
        y = object['topleft']['y']
        h = object['bottomright']['y'] - object['topleft']['y']
        x = object['topleft']['x']
        w = object['bottomright']['x'] - object['topleft']['x']
        self.object = object
        self.track_window = (x, y, w, h)
        self.yolo = object
        self.currentcount = 0
        center = getcenter(self.track_window)
        self.distance = getdistance(center, frame)

        # set up the ROI for tracking
        self.roi = frame[y:y+h, x:x+w]
        self.hsv_roi = cv2.cvtColor(self.roi, cv2.COLOR_RGB2HSV)

        self.maskinv = cv2.inRange(self.hsv_roi, np.array((0, 0, 0)), np.array((180, 255, 160)))
        self.mask = cv2.bitwise_not(self.maskinv)

        self.roi_hist = cv2.calcHist([self.hsv_roi], [0], self.maskinv, [180], [0, 180])
        cv2.normalize(self.roi_hist, self.roi_hist, 0, 255, cv2.NORM_MINMAX)

    def getcurrenttrackwindow(self):
        return self.track_window

    def setcurrenttrackwindow(self,window):
        self.track_window = window

    def getcenterhistory(self):
        return self.trackhistory

    def getcurrentcount(self):
        return self.currentcount

    def setcurrentcount(self, count):
        self.currentcount = count

    def getyoloparameters(self):
        return self.yolo

    #Methode zum finden der neuen Position mithilfe der bestehenden Region of Interest
    def nextimage(self, frame):
        self.currentcount = self.currentcount + 1

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv], [0], self.roi_hist, [0, 180], 1)

        # apply meanshift to get the new location
        ret, newtrack_window = cv2.meanShift(dst, self.track_window, self.term_crit)
        center = getcenter(newtrack_window)
        self.distance = getdistance(center, frame)
        self.trackhistory.append(center)
        self.track_window = newtrack_window
        stop = False
        if self.maxcount < self.currentcount:
            stop = True

        return stop

    #Methode zum Einzeichnen im Cockpit
    def drawobjectBB(self, frame):
        if self.currentcount > 0:
            outputstring = "ID: " + str(self.IDobject) +' '+  self.yolo['label'] + "_MEANSHIFT_d: " + str(self.distance) + " Meter"
            x,y,w,h = self.track_window
        else:
            outputstring = "ID: " + str(self.IDobject) +' '+ self.yolo['label'] + " d: "+ str(self.distance) + " Meter"
            x = self.yolo['topleft']['x']
            y = self.yolo['topleft']['y']
            w = self.yolo['bottomright']['x'] - self.yolo['topleft']['x']
            h = self.yolo['bottomright']['y'] - self.yolo['topleft']['y']

        labelcolor = (255,255,255)
        if self.yolo['label']=='person':
           labelcolor = (82, 34, 139)

        elif self.yolo['label']=='train':
           labelcolor = (127, 255, 0)

        elif self.yolo['label']=='car':
            labelcolor = (0, 238, 0)
        elif self.yolo['label']=='motorcycle':
            labelcolor = (0, 255, 127)

        cv2.putText(frame,outputstring,(x,y-5), cv2.FONT_HERSHEY_PLAIN, 1 ,labelcolor, 2,cv2.LINE_AA)
        cv2.rectangle(frame, (x,y), (x+w,y+h), labelcolor,2)



#Ermittelt das Zentrum einer Bounding Box
def getcenter(window):
    center_x = int(round(window[0] + window[2] / 2,0))
    center_y = int(round(window[1] + window[3] / 2,0))
    return center_x, center_y

#TEST-Methode zur Ermittlung der Entfernung anhand der x und y-Koordinate
def getdistance(point, frame):
    height, width, depth = frame.shape
    xpercent = point[0]/width
    ypercent = (height-point[1])/height

    if xpercent > 0.5:
        xpercent = xpercent-0.5
        if -2.67*xpercent + 0.8 < ypercent: #Punkt liegt auf der rechten Bildhälfte außerhalb des Schienenbetts
            yanschiene = max(-2.67*xpercent + 0.8,0)
            dist = yanschiene*yanschiene*yanschiene*200
        else: #Punkt liegt auf der rechten Bildhälfte innerhalb des Schienenbetts
            dist = ypercent*ypercent*ypercent*200
    else:
        if 1.14*xpercent +0.23 > ypercent: #Punkt liegt auf der linken Seite im Schienenbett
            dist = ypercent*ypercent*ypercent*200
        else: #Punkt liegt auf der linken Bildhälfte außerhalb des Schienenbetts
            yanschiene = 1.14*xpercent +0.23
            dist = yanschiene*yanschiene*yanschiene*200

    dist = round(dist,0)
    return dist





'''
MOSSE tracking sample

This sample implements correlation-based tracking approach, described in [1].

Usage:
  mosse.py [--pause] [<video source>]

  --pause  -  Start with playback paused at the first video frame.
              Useful for tracking target selection.

  Draw rectangles around objects with a mouse to track them.

Keys:
  SPACE    - pause video
  c        - clear targets

[1] David S. Bolme et al. "Visual Object Tracking using Adaptive Correlation Filters"
    http://www.cs.colostate.edu/~bolme/publications/Bolme2010Tracking.pdf
'''

# Python 2/3 compatibility
#from __future__ import print_function
import sys
PY3 = sys.version_info[0] == 3

if PY3:
    xrange = range

import numpy as np
import cv2
from common import draw_str, RectSelector
import video

def rnd_warp(a):
    h, w = a.shape[:2]
    T = np.zeros((2, 3))
    coef = 0.2
    ang = (np.random.rand()-0.5)*coef
    c, s = np.cos(ang), np.sin(ang)
    T[:2, :2] = [[c,-s], [s, c]]
    T[:2, :2] += (np.random.rand(2, 2) - 0.5)*coef
    c = (w/2, h/2)
    T[:,2] = c - np.dot(T[:2, :2], c)
    return cv2.warpAffine(a, T, (w, h), borderMode = cv2.BORDER_REFLECT)

def divSpec(A, B):
    Ar, Ai = A[...,0], A[...,1]
    Br, Bi = B[...,0], B[...,1]
    C = (Ar+1j*Ai)/(Br+1j*Bi)
    C = np.dstack([np.real(C), np.imag(C)]).copy()
    return C

eps = 1e-5

class MOSSE:
    def __init__(self, frame, rect, count):
        x1, y1, x2, y2 = rect
        w, h = map(cv2.getOptimalDFTSize, [x2-x1, y2-y1])
        x1, y1 = (x1+x2-w)//2, (y1+y2-h)//2
        self.pos = x, y = x1+0.5*(w-1), y1+0.5*(h-1)
        self.size = w, h
        self.maxcount = count
        self.currentcount = 0
        img = cv2.getRectSubPix(frame, (w, h), (x, y))

        self.win = cv2.createHanningWindow((w, h), cv2.CV_32F)
        g = np.zeros((h, w), np.float32)
        g[h//2, w//2] = 1
        g = cv2.GaussianBlur(g, (-1, -1), 2.0)
        g /= g.max()

        self.G = cv2.dft(g, flags=cv2.DFT_COMPLEX_OUTPUT)
        self.H1 = np.zeros_like(self.G)
        self.H2 = np.zeros_like(self.G)
        for i in xrange(128):
            a = self.preprocess(rnd_warp(img))
            A = cv2.dft(a, flags=cv2.DFT_COMPLEX_OUTPUT)
            self.H1 += cv2.mulSpectrums(self.G, A, 0, conjB=True)
            self.H2 += cv2.mulSpectrums(     A, A, 0, conjB=True)
        self.update_kernel()
        self.update(frame)


    def update(self, frame, rate = 0.125):

        if self.maxcount > self.currentcount:
            (x, y), (w, h) = self.pos, self.size
            self.last_img = img = cv2.getRectSubPix(frame, (w, h), (x, y))
            img = self.preprocess(img)
            self.last_resp, (dx, dy), self.psr = self.correlate(img)
            self.good = self.psr > 4.0  #8.0
            self.currentcount = self.currentcount + 1

            if not self.good:
                print("NOTGOOD")
                return False

            self.pos = x+dx, y+dy
            self.last_img = img = cv2.getRectSubPix(frame, (w, h), self.pos)
            img = self.preprocess(img)

            A = cv2.dft(img, flags=cv2.DFT_COMPLEX_OUTPUT)
            H1 = cv2.mulSpectrums(self.G, A, 0, conjB=True)
            H2 = cv2.mulSpectrums(     A, A, 0, conjB=True)
            self.H1 = self.H1 * (1.0-rate) + H1 * rate
            self.H2 = self.H2 * (1.0-rate) + H2 * rate
            self.update_kernel()
        else:
            return True

    @property
    def state_vis(self):
        f = cv2.idft(self.H, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT )
        h, w = f.shape
        f = np.roll(f, -h//2, 0)
        f = np.roll(f, -w//2, 1)
        kernel = np.uint8( (f-f.min()) / f.ptp()*255 )
        resp = self.last_resp
        resp = np.uint8(np.clip(resp/resp.max(), 0, 1)*255)
        vis = np.hstack([self.last_img, kernel, resp])
        return vis

    def draw_state(self, vis):
        (x, y), (w, h) = self.pos, self.size
        x1, y1, x2, y2 = int(x-0.5*w), int(y-0.5*h), int(x+0.5*w), int(y+0.5*h)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255))
        if self.good:
            cv2.circle(vis, (int(x), int(y)), 2, (0, 0, 255), -1)
        else:
            cv2.line(vis, (x1, y1), (x2, y2), (0, 0, 255))
            cv2.line(vis, (x2, y1), (x1, y2), (0, 0, 255))
        draw_str(vis, (x1, y2+16), 'PSR: %.2f' % self.psr)

    def preprocess(self, img):
        img = np.log(np.float32(img)+1.0)
        img = (img-img.mean()) / (img.std()+eps)
        return img*self.win

    def correlate(self, img):
        C = cv2.mulSpectrums(cv2.dft(img, flags=cv2.DFT_COMPLEX_OUTPUT), self.H, 0, conjB=True)
        resp = cv2.idft(C, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
        h, w = resp.shape
        _, mval, _, (mx, my) = cv2.minMaxLoc(resp)
        side_resp = resp.copy()
        cv2.rectangle(side_resp, (mx-5, my-5), (mx+5, my+5), 0, -1)
        smean, sstd = side_resp.mean(), side_resp.std()
        psr = (mval-smean) / (sstd+eps)
        return resp, (mx-w//2, my-h//2), psr

    def update_kernel(self):
        self.H = divSpec(self.H1, self.H2)
        self.H[...,1] *= -1

class App:
    def __init__(self, video_src, paused = False):
        self.cap = video.create_capture(video_src)
        _, self.frame = self.cap.read()
        cv2.imshow('frame', self.frame)
        self.rect_sel = RectSelector('frame', self.onrect)
        self.trackers = []
        self.paused = paused


    def nextimage(self):
        while True:
            if not self.paused:
                ret, self.frame = self.cap.read()
                if not ret:
                    break
                frame_gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
                for tracker in self.trackers:
                    tracker.update(frame_gray)

            vis = self.frame.copy()
            for tracker in self.trackers:
                tracker.draw_state(vis)
            if len(self.trackers) > 0:
                cv2.imshow('tracker state', self.trackers[-1].state_vis)
            self.rect_sel.draw(vis)

            cv2.imshow('frame', vis)
            ch = cv2.waitKey(10)
            if ch == 27:
                break
            if ch == ord(' '):
                self.paused = not self.paused
            if ch == ord('c'):
                self.trackers = []


