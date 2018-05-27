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
                "Object_ID": id+1,
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
                "Object_ID": id+1,
                "topleft": {
                    "x": box[0],
                    "y": box[1]},
                "bottomright": {
                    "x": box[2],
                    "y": box[3]}
            })

def synthesize(anzahl):
    annotations = []
    for a in range(0,anzahl):
        objects = list()



        #Hintergrund wählen
        image = su.getbackground()
        height,width,colors = image.shape
        print(image.shape)

        imagemask = np.zeros((height, width), dtype=int)

        print(imagemask.shape)

        for b in range(0,sto.putBees()):
            #Bienenbild wählen
            original, mask, replaced, label = su.getonlinebee()
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
                    #op, mp = su.flip(op, mp,'POLLEN')
                    #op, mp = su.rotate(op, mp,'POLLEN')
                    #op, mp = su.resize(op, mp,'POLLEN')

                    #Polle einzeichnen
                    original, mask, box = su.placePolle(original, mask, op, mp)

                    savepath = PATH.DATAPATH + "SynthTrainingData/"
                    print("Saving image")
                    cv2.imwrite(savepath + str(a) + "_original.jpg", original)
                    cv2.imwrite(savepath + str(a) + "_mask.jpg", mask)

                    #addannotation(objects, annotations, box, lp, a, c)


            #Milbe
            if False: #sto.putMite():
                print("putting Mite")
                #ChooseMite
                om, mm, rm, lm = su.getMite()
                #ManipulateMite
                om, mm = su.flip(om, mm,'MITE')
                om, mm = su.rotate(om, mm,'MITE')
                om, mm = su.resize(om, mm,'MITE')

                #AddMite
                original, mask, box = su.placeMite(original, mask, om, mm)
                addannotation(objects, annotations, box, lm, a, c)

            # #Gesamtbienenbild manipulieren
            # original, mask = su.flip(original, mask, 'BEE')
            # original, mask = su.rotate(original, mask, 'BEE')
            # original, mask = su.resize(original, mask, 'BEE')
            #
            #
            # #Biene einzeichnen
            # image, box = su.placeBee(image, original, mask, objects)
            # addannotation(objects, annotations, box, label, a, b)


        savepath = PATH.DATAPATH + "SynthTrainingData/"
        #print("Saving image")
        #cv2.imwrite(savepath + str(a) + "_synth.jpg", image)
        #bbox = su.drawBBox(image, objects)
        #cv2.imwrite(savepath + str(a) + "_synth_bbox.jpg", bbox)

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
