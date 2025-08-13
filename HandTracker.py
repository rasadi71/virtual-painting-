# /home/rasadi/Desktop/python_scripts/handTracker.py
import cv2
import mediapipe as mp

class HandTracker:
    def __init__(self, detectionCon=0.5):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(min_detection_confidence=detectionCon)
        self.mp_draw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)
        if draw and self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(img, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
        return img

    def getPosition(self, img, draw=False):
        positions = []
        if self.results.multi_hand_landmarks:  # Corrected from multi_usbwc
            for hand_landmarks in self.results.multi_hand_landmarks:
                for id, lm in enumerate(hand_landmarks.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    positions.append((cx, cy))
        return positions

    def getUpFingers(self, img):
        up_fingers = [0] * 5  # Thumb, Index, Middle, Ring, Pinky
        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                up_fingers[1] = 1 if hand_landmarks.landmark[8].y < hand_landmarks.landmark[6].y else 0  # Index
                up_fingers[2] = 1 if hand_landmarks.landmark[12].y < hand_landmarks.landmark[10].y else 0  # Middle
        return up_fingers
