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

