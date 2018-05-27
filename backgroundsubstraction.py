import os
import cv2
import numpy as np

os.chdir('/Users/Julian/Desktop/Dropbox/synthbeedata/Raw/')

cap = cv2.VideoCapture('output_2.mp4')

fgbg = cv2.createBackgroundSubtractorMOG2()

start = 250
end =  int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.set(1, start)

for i in range(0, end-start):
    ret, frame = cap.read()

    #Hintergrundsubtraktion
    frame = fgbg.apply(frame)


    #VORVERARBEITUNG
    #Bilateralfilter
    medianblur = cv2.medianBlur(frame,5)

    bilateral = cv2.bilateralFilter(medianblur,20,75,75)


    #binary = cv2.adaptiveThreshold(bilateral, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 4)
    _, binary = cv2.threshold(bilateral, 100, 255, cv2.THRESH_BINARY)

    #Mophologie
    #kernel = np.ones((5,5),np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))

    binarydilated = cv2.dilate(binary,kernel,iterations = 3)
    binaryclosed = cv2.erode(frame,kernel,iterations = 1)





    binarybilateral = cv2.adaptiveThreshold(binary, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)


    #MEDIANFILTER
    #medianblur = cv2.medianBlur(binarybilateral,11)

    #frame = cv2.blur(frame,(30,30))




    cv2.namedWindow('Backgroundsubstraction', flags=cv2.WINDOW_NORMAL)
    cv2.imshow('Backgroundsubstraction', frame)
    # cv2.namedWindow('Medianfilter', flags=cv2.WINDOW_NORMAL)
    # cv2.imshow('Medianfilter', medianblur)
    # cv2.namedWindow('Bilateralfilter', flags=cv2.WINDOW_NORMAL)
    # cv2.imshow('Bilateralfilter', bilateral)

    cv2.namedWindow('Binary', flags=cv2.WINDOW_NORMAL)
    cv2.imshow('Binary', binary)

    cv2.namedWindow('adaptiveThreshold', flags=cv2.WINDOW_NORMAL)
    cv2.imshow('adaptiveThreshold', binarybilateral)

    cv2.namedWindow('Dil+Erosion', flags=cv2.WINDOW_NORMAL)
    cv2.imshow('Dil+Erosion', binaryclosed)

    cv2.namedWindow('Dilatation', flags=cv2.WINDOW_NORMAL)
    cv2.imshow('Dilatation', binarydilated)

    # cv2.namedWindow('adaptiveThresholdbilateral', flags=cv2.WINDOW_NORMAL)
    # cv2.imshow('adaptiveThresholdbilateral', binarybilateral)
    #

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()


file1 = 'VideoNr_0_frame_87.jpg'
file2 = 'background.jpg'

frame1 = cv2.imread(file1)
background = cv2.imread(file2)


