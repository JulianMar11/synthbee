import synthesizeutil as su
import cv2
import numpy as np
import csv

#Standards
width = 350
height = 340

import PATH
import synthesizestochastic as sto
print(PATH.DATAPATH)


def addannotation(objects, annotations, box, label, frame, id):
    objects.append({
                "label": str(label),
                "Frame_ID":   frame+1,
                "Bee_ID": id+1,
                "topleft": {
                    "x": box[0],
                    "y": box[1]},
                "bottomright": {
                    "x": box[2],
                    "y": box[3]}
            })
    annotations.append({
                "label": str(label),
                "Frame_ID":  frame+1,
                "Bee_ID": id+1,
                "topleft": {
                    "x": box[0],
                    "y": box[1]},
                "bottomright": {
                    "x": box[2],
                    "y": box[3]}
            })

def addbeeannotation(list, box, label, frame, beeid):
    list.append({
        "label": str(label),
        "Frame_ID": frame + 1,
        "Bee_ID": beeid + 1,
        "topleft": {
            "x": box[0],
            "y": box[1]},
        "bottomright": {
            "x": box[2],
            "y": box[3]}
    })

def copybeeparams(beeparams, annotations, objects):
    for object in beeparams:
        annotations.append({
                "label": object['label'],
                "Frame_ID":  object['Frame_ID'],
                "Bee_ID": object['Bee_ID'],
                "topleft": {
                    "x": object['topleft']['x'],
                    "y": object['topleft']['y']},
                "bottomright": {
                    "x": object['bottomright']['x'],
                    "y": object['bottomright']['y']}
            })
        objects.append({
                "label": object['label'],
                "Frame_ID":  object['Frame_ID'],
                "Bee_ID": object['Bee_ID'],
                "topleft": {
                    "x": object['topleft']['x'],
                    "y": object['topleft']['y']},
                "bottomright": {
                    "x": object['bottomright']['x'],
                    "y": object['bottomright']['y']}
            })

def synthesize(anzahl):
    annotations = []
    objid = 0
    savepath = PATH.DATAPATH + "SynthTrainingData/"

    for a in range(0,anzahl):
        objects = list()

        #Hintergrund wählen
        image = su.getbackground()
        height,width,colors = image.shape
        imagemask = np.zeros((height, width), dtype=int)


        for b in range(0,sto.putBees()):
            #Bienenbild wählen
            original, mask, replaced, label = su.getonlinebee()
            beeparams = []

            # Gesamtbienenbild manipulieren
            original, mask = su.flip(original, mask, 'BEE')
            original, mask = su.rotate(original, mask, 'BEE')
            #original, mask = su.resize(original, mask, 'BEE')

            _ , mask = cv2.threshold(mask,127,255,cv2.THRESH_BINARY)
            print(mask.shape)

            #Pollen
            if sto.putPoll():
                anz = sto.anzPolls()
                for c in range(0,anz):
                    print("putting Pollen")
                    #Pollenbild wählen
                    op, mp, rp, lp = su.getPollen()
                    print(mp.shape)

                    #Bild manipulieren
                    op, mp = su.flip(op, mp,'POLLEN')
                    op, mp = su.rotate(op, mp,'POLLEN')
                    op, mp = su.resize(op, mp,'POLLEN')
                    _, mp = cv2.threshold(mp,127,255,cv2.THRESH_BINARY)

                    #Polle einzeichnen
                    originalnew, masknew, box, exception = su.place_Parzipolle(original, mask, op, mp)
                    if not exception:
                        original = originalnew
                        mask = masknew
                        cv2.imwrite(savepath + str(a) + "_original.jpg", original)
                        cv2.imwrite(savepath + str(a) + "_mask.jpg", mask)
                        addbeeannotation(beeparams, box, lp, a, b)

            #Milbe
            if sto.putMite():
                print("putting Mite")
                #ChooseMite
                om, mm, rm, lm = su.getMite()
                #ManipulateMite
                om, mm = su.flip(om, mm,'MITE')
                om, mm = su.rotate(om, mm,'MITE')
                om, mm = su.resize(om, mm,'MITE')
                _, mm = cv2.threshold(mm,127,255,cv2.THRESH_BINARY)

                #AddMite
                originalnew, masknew, box, exception = su.place_Mite(original, mask, om, mm)
                if not exception:
                    original = originalnew
                    mask = masknew
                    cv2.imwrite(savepath + str(a) + "_original.jpg", original)
                    cv2.imwrite(savepath + str(a) + "_mask.jpg", mask)
                    addbeeannotation(beeparams, box, lm, a, b)

            #Biene einzeichnen
            image, box, correctbeeparams = su.placeBee(image, original, mask, objects, beeparams)
            beeparams = []
            copybeeparams(correctbeeparams, annotations,objects)
            addannotation(objects, annotations, box, label, a, b)



        print("Saving image")
        cv2.imwrite(savepath + str(a) + "_synth.jpg", image)
        bbox = su.drawBBox(image, objects)
        cv2.imwrite(savepath + str(a) + "_synth_bbox.jpg", bbox)

    return annotations



###Auswertungsdateien speichern
def saveresults(annotations):
    print("Annotationen speichern")
    keys = annotations[0].keys()
    print(keys)
    with open('annotations.csv', 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=keys)
        writer.writeheader()
        for p in annotations:
            writer.writerow(p)


results = synthesize(20)
print(results)
saveresults(results)
