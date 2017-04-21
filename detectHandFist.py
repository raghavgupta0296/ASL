import cv2
import os
from PIL import Image
import testCNN

# alphabet = 'A'
# ctr = '0'
# path = os.path.join("TrainingImages",alphabet)
# if not os.path.isdir(path):
#     os.mkdir(path)

classifier = cv2.CascadeClassifier("hand2.xml")
classifier2 = cv2.CascadeClassifier("hand.xml")
cam = cv2.VideoCapture(0)

t= testCNN.testing()

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
        hand = frame[y:y+300,x:x+300]
        # resizedFrame = cv2.resize(hand,(112,92))
        cv2.rectangle(frame, (x, y), (x + 300, y + 300), (204, 153, 255), 2)
        if key == 32:
            # cv2.imwrite("%s/%s.png" % (path, ctr), hand)
            # ctr = str(int(ctr)+1)
            cv2.imwrite("20.png",frame[y:y+300,x:x+300])
            t.test_im(frame[y:y+300,x:x+300])

    elif fists:
        fist_i = fists[0]
        (x,y,w,h) = [v*4 for v in fist_i]
        fist = frame[y:y+300,x:x+300]
        # resizedFrame = cv2.resize(fist, (112, 92))
        cv2.rectangle(frame, (x, y), (x + 300, y + 300), (204, 153, 255), 2)
        if key == 32:
            # cv2.imwrite("%s/%s.png" % (path, ctr), fist)
            # ctr = str(int(ctr) + 1)
            cv2.imwrite("tested.png",frame[y:y+300,x:x+300])
            t.test_im(frame[y:y+300,x:x+300])

    cv2.imshow("webcam",frame)




