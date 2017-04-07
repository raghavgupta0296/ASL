import cv2

cam = cv2.VideoCapture(0)
i=0
num = 1

while cam.isOpened:
    key = cv2.waitKey(10)
    if key==27:
        break
    r, frame = cam.read()
    frame = cv2.flip(frame,1,0)
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    if(i==0):
        x = int(frame.shape[1]/2)
        y = int(frame.shape[0]/2)

    cv2.rectangle(frame,(x-150,y-150),(x+150,y+150),2)
    cv2.imshow("store pic",frame)
    if key == 32:
        cv2.imwrite("./TrainingImages/B/%s.png"%num,frame[y-150:y+150,x-150:x+150])
        num+=1
    i+=1
