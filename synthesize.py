import synthesizeutil as su
import cv2
import numpy as np
import csv

#Standards
width = 350
height = 340

import PATH

print(PATH.DATAPATH)



def synthesize(anzahl):
    annotations = []
    for a in range(0,anzahl):
        objects = list()

        #Hintergrund wählen
        background = su.getbackground()
        #Anzahl Bienen pro Bild
        r = np.random.random_integers(1,4,1)[0]

        for i in range(0,r):
            #Bienenbild wählen
            original, mask, replaced, label = su.getonlinebee()
            #Bild manipulieren
            original, mask, replaced = su.flip(original, mask, replaced)
            original, mask, replaced = su.rotate(original, mask, replaced)
            original, mask, replaced = su.resize(original, mask, replaced)


            #Biene einzeichnen
            background, box = su.placebee(background, original, mask, objects)
            objects.append({
                "label": str(label),
                "Frame_ID":   a+1,
                "Object_ID": i+1,
                "topleft": {
                    "x": box[0],
                    "y": box[1]},
                "bottomright": {
                    "x": box[2],
                    "y": box[3]}
            })
            annotations.append({
                "label": str(label),
                "Frame_ID":  a+1,
                "Object_ID": i+1,
                "topleft": {
                    "x": box[0],
                    "y": box[1]},
                "bottomright": {
                    "x": box[2],
                    "y": box[3]}
            })

        savepath = PATH.DATAPATH + "SynthTrainingData/"
        print("Saving image")
        cv2.imwrite(savepath + str(a) + "_synth.jpg", background)
        bbox = su.drawBBox(background, objects)
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


results = synthesize(500)
print(results)
saveresults(results)
