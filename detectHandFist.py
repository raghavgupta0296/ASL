import cv2
import os
from PIL import Image

alphabet = 'A'
ctr = '0'
path = os.path.join("TrainingImages",alphabet)
if not os.path.isdir(path):
    os.mkdir(path)

classifier = cv2.CascadeClassifier("hand2.xml")
classifier2 = cv2.CascadeClassifier("hand.xml")
cam = cv2.VideoCapture(0)

while cam.isOpened:
    key = cv2.waitKey(10)
    if key == 27:
        break
    r,frame = cam.read()
    cv2.flip(frame,1,0)
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    miniFrame = cv2.resize(frame,(int(frame.shape[1]/4),int(frame.shape[0]/4)))
    hands = classifier.detectMultiScale(miniFrame)
    fists = classifier2.detectMultiScale(miniFrame)
    hands = sorted(hands, key=lambda x: x[3])
    fists = sorted(fists, key=lambda x: x[3])
    if hands:
        hand_i = hands[0]
        (x,y,w,h) = [v*4 for v in hand_i]
        hand = frame[y:y+400,x:x+400]
        # resizedFrame = cv2.resize(hand,(112,92))
        cv2.rectangle(frame, (x, y), (x + 400, y + 400), (204, 153, 255), 2)
        if key == 32:
            cv2.imwrite("%s/%s.png" % (path, ctr), hand)
            ctr = str(int(ctr)+1)


    elif fists:
        fist_i = fists[0]
        (x,y,w,h) = [v*4 for v in fist_i]
        fist = frame[y:y+400,x:x+400]
        # resizedFrame = cv2.resize(fist, (112, 92))
        cv2.rectangle(frame, (x, y), (x + 400, y + 400), (204, 153, 255), 2)
        if key == 32:
            cv2.imwrite("%s/%s.png" % (path, ctr), fist)
            ctr = str(int(ctr) + 1)

    cv2.imshow("webcam",frame)



