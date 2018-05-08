import cv2
import numpy as np
from imutils.object_detection import non_max_suppression

### RENDERER Configuration
imagecount = 0


activeyolo = True
activetracker = False
activeMosse = True

#### OBJECT DETECTION
if activeyolo:
    #Import YOLO
    import sys
    sys.path.append('/Users/Julian/GitHub/darkflow-master')
    print("YOLO - Building CNN")
    import yolo as persondetection
    import csv
    #Parameters YOLO
    IDobject = 0



#### OBJECT TRACKER
if activetracker:
    #Import TRACKER
    import tracker as tracking
    #Parameters TRACKER
    objecttracks = []


#### Mosse TRACKER
if activeMosse:
    #Import TRACKER
    import Mossetracker as tracking
    #Parameters TRACKER
    objecttracks = []


def renderimage(image):
    #Globale Variablenzugriffe
    global imagecount
    global IDobject

    print("RENDERER - Starting frame " + str(imagecount))

    #Erzeugt Resultatbild
    result = image.copy()

    if activetracker:
        #Bestehende Objekttracker aktualisieren
        print("TRACKER - Update current object trackers")
        for object in objecttracks:
            stop = object.nextimage(image)
            if stop:
                objecttracks.remove(object)
                print("TRACKER - Remove outdated object tracker")

    if activeMosse:
        #Bestehende Mossetracker aktualisieren
        print("TRACKER - Update current object trackers")
        for object in objecttracks:
            frame_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            stop = object.update(frame_gray)
            if stop:
                objecttracks.remove(object)
                print("TRACKER - Remove outdated object tracker")




    if activeyolo:
        ##YOLO
        print("YOLO - Detecting new objects")
        YOLO_objects = persondetection.YOLOperson(image)
        #Annotiert aktuelle Framenummer
        for box in YOLO_objects:
            box['image'] = imagecount

        #Compare to currently tracked objects
        if activetracker:
            remainingdetects = []
            for detect in YOLO_objects:
                cancel = False
                x = detect['topleft']['x']
                y = detect['topleft']['y']
                w = detect['bottomright']['x'] - detect['topleft']['x']
                h = detect['bottomright']['y'] - detect['topleft']['y']
                for track in objecttracks:
                    currentwindow = track.getcurrenttrackwindow()
                    xt,yt,wt,ht = currentwindow
                    detectandtrack = np.array([[x,y, x+w,y+h],[xt, yt, xt + wt, yt + ht]])
                    suppression = non_max_suppression(detectandtrack, probs=None, overlapThresh=0.4)
                    if len(suppression)==1:
                        cancel = True
                        print("YOLO - Removing new detection due to existing person tracker")
                        track.update(detect, image)

                if not cancel:
                    remainingdetects.append(detect)
            YOLO_objects = remainingdetects

            #Initialisiert neue Objekttracker
            for object in YOLO_objects:
                IDobject = IDobject + 1
                object['ID'] = IDobject
                newtracker = tracking.trackobject(10, object, image)
                objecttracks.append(newtracker)

        if activeMosse:
            remainingdetects = []
            for detect in YOLO_objects:
                cancel = False
                x = detect['topleft']['x']
                y = detect['topleft']['y']
                w = detect['bottomright']['x'] - detect['topleft']['x']
                h = detect['bottomright']['y'] - detect['topleft']['y']
                for track in objecttracks:
                    (xt, yt), (wt, ht) = track.pos, track.size
                    x1, y1, x2, y2 = int(xt-0.5*wt), int(yt-0.5*ht), int(xt+0.5*wt), int(yt+0.5*ht)
                    detectandtrack = np.array([[x,y, x+w,y+h],[x1, y1, x2, y2]])
                    suppression = non_max_suppression(detectandtrack, probs=None, overlapThresh=0.8)
                    if len(suppression)==1:
                        cancel = True
                        print("TRACKER - Removing current tracker due to existing new detection")
                        objecttracks.remove(track)

                if not cancel:
                    remainingdetects.append(detect)

            #Initialisiert neue Objekttracker
            for object in YOLO_objects:
                #IDobject = IDobject + 1
                #object['ID'] = IDobject
                x1 = object['topleft']['x']
                y1 = object['topleft']['y']
                x2 = object['bottomright']['x']
                y2 = object['bottomright']['y']
                frame_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                rect = (x1,y1,x2,y2)
                newtracker = tracking.MOSSE(frame_gray, rect, 15)
                objecttracks.append(newtracker)

    ##Ausgabe kreieren
    print("RENDERER - Create Output")
    if activetracker:
        print("TRACKER - Drawing Bounding Boxes")
        #Bestehende Objekttracker einzeichnen
        lastsignal = []
        for object in objecttracks:
            object.drawobjectBB(result)

    if activeMosse:
        print("Mosse - Drawing Bounding Boxes")

        for tracker in objecttracks:
                tracker.draw_state(result)

    imagecount = imagecount + 1
    return result


def checktext(text,previoustext):
    check = False
    if not text == 'unbekannt':
        if len(previoustext) <= len(text):
            check = True
    return check


###Auswertungsdateien speichern
def saveresults():
    print("RENDERER - Ergebnisse speichern")
    keys = objecttracks[0].keys()
    with open('detections.csv', 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=keys)
        writer.writeheader()
        for p in objecttracks:
            writer.writerow(p)
