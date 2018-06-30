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
                "Frame_ID":   frame,
                "Bee_ID": id,
                "topleft": {
                    "x": box[0],
                    "y": box[1]},
                "bottomright": {
                    "x": box[2],
                    "y": box[3]}
            })
    annotations.append({
                "label": str(label),
                "Frame_ID":  frame,
                "Bee_ID": id,
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
        "Frame_ID": frame,
        "Bee_ID": beeid,
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

def saveoutput(beeparams, a):
    string = ""
    for object in beeparams:
        labelinfo = 3
        if str(object['label']) == "Biene":
            labelinfo = 0
        elif str(object['label']) == "Polle":
            labelinfo = 1
        elif str(object['label']) == "Milbe":
            labelinfo = 2

        boxinfo = " " + str(object['topleft']['y']) + "," + str(object['topleft']['x']) + "," + str(object['bottomright']['y']) + "," + str(object['bottomright']['x'])
        string = string + boxinfo + ',' + str(labelinfo)

    path = "../neuronalnet/output/" + str(a) + ".jpg"
    #../neuronalnet/output/5_1_179_3.png 0,0,212,190,0
    finalstring = path + string + '\n'
    #print(finalstring)
    return finalstring

def synthesize(anzahl):
    annotations = []
    output = []
    objid = 0
    savepath = PATH.DATAPATH + "SynthTrainingData/"

    for a in range(1,1 + anzahl):
        print("Synthesise picture" + str(a))
        objects = list()

        #Hintergrund wählen
        image = su.getbackground()
        height,width,colors = image.shape

        for b in range(0,sto.putBees()):
            #Bienenbild wählen
            original, mask, replaced, label, exe = su.getbee()
            beeparams = []
            if not exe:
                # Gesamtbienenbild manipulieren
                original, mask = su.manipulate(original, mask, 'BEE')

                #Pollen
                if sto.putPoll():
                    anz = sto.anzPolls()
                    for c in range(0,anz):
                        #print("putting Pollen")
                        #Pollenbild wählen

                        op, mp, rp, lp, ex = su.getPollen()
                        if not ex:
                            #Bild manipulieren
                            op, mp = su.manipulate(op, mp, 'POLLEN')

                            #Polle einzeichnen
                            originalnew, masknew, box, exception = su.place_Parzipolle(original, mask, op, mp)
                            if not exception:
                                original = originalnew
                                mask = masknew
                                #cv2.imwrite(savepath + str(a) + "_original.jpg", original)
                                #cv2.imwrite(savepath + str(a) + "_mask.jpg", mask)
                                addbeeannotation(beeparams, box, lp, a, b)

                #Milbe
                if sto.putMite():
                    #print("putting Mite")
                    #ChooseMite
                    om, mm, rm, lm, ex = su.getMite()

                    if not ex:
                        #ManipulateMite
                        om, mm = su.manipulate(om, mm, 'MITE')

                        #AddMite
                        originalnew, masknew, box, exception = su.place_Mite(original, mask, om, mm)
                        if not exception:
                            original = originalnew
                            mask = masknew
                            #cv2.imwrite(savepath + str(a) + "_original.jpg", original)
                            #cv2.imwrite(savepath + str(a) + "_mask.jpg", mask)
                            addbeeannotation(beeparams, box, lm, a, b)

                #Biene einzeichnen
                imagenew, box, correctbeeparams, beeexe = su.placeBee(image, original, mask, objects, beeparams)
                beeparams = []
                if not ex:
                    image = imagenew
                    copybeeparams(correctbeeparams, annotations,objects)
                    addannotation(objects, annotations, box, label, a, b)



        cv2.imwrite(savepath + str(a) + ".jpg", image)
        output.append(saveoutput(objects, a))
        #bbox = su.drawBBox(image, objects)
        #cv2.imwrite(savepath + str(a) + "_bbox.jpg", bbox)
        if a%200 == 0:
            saveresults(annotations, output)

    return annotations, output



###Auswertungsdateien speichern
def saveresults(annotations, output):
    print("Output und Annotation speichern")

    # Open a file
    fo = open("output.txt", "w")
    # Write sequence of lines at the end of the file.
    line = fo.writelines(output)

    # Close opend file
    fo.close()
    #np.savetxt('test.out', output, delimiter=';')

    keys = annotations[0].keys()
    with open('annotations.csv', 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=keys)
        writer.writeheader()
        for p in annotations:
            writer.writerow(p)


results, resultsout = synthesize(10000)
saveresults(results, resultsout)
