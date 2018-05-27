import cv2
import os
import render as ren


def playvideo(filenumber, end, savevideo, saveframes, savedata):
    os.chdir('/Users/Julian/Desktop/Dropbox/synthbeedata/Raw/')
    dir_path = os.path.dirname(os.path.realpath(__file__))

    files = [
        'output_2.mp4',
    ]

    #1.Spalte = Offset   2.Spalte = Wieviele Frame 3.Spalte = Jump
    variables = [
            [0, 0, 1]
        ]

    cap = cv2.VideoCapture('/Users/Julian/Desktop/Dropbox/synthbeedata/'+files[filenumber])
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if variables[filenumber][0] < length:
        cap.set(1, variables[filenumber][0])

    # Default resolutions of the frame are obtained.The default resolutions are system dependent.
    # We convert the resolutions from float to integer.
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    print(frame_height, frame_width)

    # Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
    if savevideo:
        out = cv2.VideoWriter('/Users/Julian/Desktop/Dropbox/synthbeedata/' + files[filenumber] + '_detected.avi',
                                  cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (frame_width, frame_height))
    if end == 0:
         end = max(length, variables[filenumber][1])

    jump = variables[filenumber][2]
    num = 0
    for i in range(0, end):
        num = num + 1
        nextframe = num * jump + variables[filenumber][0]
        cap.set(1, nextframe)
        ret, frame = cap.read()
        if ret == True:
            frame = ren.renderimage(frame)
            # Saves for video
            if savevideo:
                out.write(frame)
            if saveframes:
                cv2.imwrite("VideoNr_" + str(filenumber) + "_frame_" + str(i) + ".jpg", frame)
            cv2.namedWindow(': ', flags=cv2.WINDOW_NORMAL)
            cv2.imshow(': ', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()

    cv2.destroyAllWindows()
    if savevideo:
        out.release()
    if savedata:
        ren.saveresults()





filenumber = 0
savevideo = False
saveframes = True
savedata = False

playvideo(filenumber, 0, savevideo, saveframes, savedata)
