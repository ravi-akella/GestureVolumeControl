import cv2
import mediapipe as mp
from time import time

class handDetector():
    def __init__(self, mode = False, maxHands = 2, detCon = 0.5, trackCon = 0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detCon = detCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.mpDraw = mp.solutions.drawing_utils
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.detCon, self.trackCon)

    def findHands(self, img, draw = True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)
        if self.results.multi_hand_landmarks:
            for handLmks in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLmks, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo = 0, draw = True):
        lmkList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lmk in enumerate(myHand.landmark):
                # print(id, lmk)
                h, w, c = img.shape
                cx, cy = int(lmk.x * w), int(lmk.y * h)
                lmkList.append([id, cx, cy])

                if draw:
                    cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)
        return lmkList



def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector()
    while True:
        _, img = cap.read()

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = detector.findHands(img)
        lmkList = detector.findPosition(img)
        if len(lmkList)!=0:
            print(lmkList[4])
        cTime = time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 3)
        cv2.imshow("HandTracking", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()





