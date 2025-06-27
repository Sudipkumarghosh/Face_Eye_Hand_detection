import cv2
from cvzone.HandTrackingModule import HandDetector

detector = HandDetector(detectionCon=int(0.8))

cap = cv2.VideoCapture(0)
cap.set(3,2120)
cap.set(4,1080)

while True:
    ret, img = cap.read()
    img= cv2.flip(img,1)
    img= detector.findHands(img)
    lmlist, bboxInfo = detector.findPosition(img)


    cv2.imshow("Image",img)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
