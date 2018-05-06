import cv2
import numpy as np
from imutils.object_detection import non_max_suppression

### RENDERER Configuration
imagecount = 0


activeyolo = True
activetracker = True

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


    ##Ausgabe kreieren
    print("RENDERER - Create Output")
    if activetracker:
        print("TRACKER - Drawing Bounding Boxes")
        #Bestehende Objekttracker einzeichnen
        lastsignal = []
        for object in objecttracks:
            object.drawobjectBB(result)


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
