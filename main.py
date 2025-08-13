# /home/rasadi/Desktop/python_scripts/main.py
import cv2
import numpy as np
import random
from handTracker import HandTracker

class ColorRect:
    def __init__(self, x, y, w, h, color, text='', alpha=0.5):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.color = color
        self.text = text
        self.alpha = alpha

    def drawRect(self, img, text_color=(255, 255, 255), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, thickness=2):
        # Draw semi-transparent rectangle
        bg_rec = img[self.y:self.y + self.h, self.x:self.x + self.w]
        # Create white_rect with the same shape and type as bg_rec, filled with self.color
        white_rect = np.zeros_like(bg_rec, dtype=np.uint8)  # Create array with same shape and type
        white_rect[:] = self.color  # Fill with color (e.g., (120, 255, 0))
        res = cv2.addWeighted(bg_rec, self.alpha, white_rect, 1 - self.alpha, 0.0)  # Changed gamma to 0.0 for simplicity
        img[self.y:self.y + self.h, self.x:self.x + self.w] = res

        # Draw text
        text_size = cv2.getTextSize(self.text, fontFace, fontScale, thickness)[0]
        text_pos = (int(self.x + self.w / 2 - text_size[0] / 2), int(self.y + self.h / 2 + text_size[1] / 2))
        cv2.putText(img, self.text, text_pos, fontFace, fontScale, text_color, thickness)

    def isOver(self, x, y):
        return self.x <= x <= self.x + self.w and self.y <= y <= self.y + self.h

# Rest of the code remains unchanged
detector = HandTracker(detectionCon=0.8)
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("Error: Could not open camera")
    exit()
cap.set(3, 1280)
cap.set(4, 720)

canvas = np.zeros((720, 1280, 3), np.uint8)
px, py = 0, 0
color = (255, 0, 0)
brushSize = 5
eraserSize = 20

colorsBtn = ColorRect(200, 0, 100, 100, (120, 255, 0), 'Colors')
colors = [
    ColorRect(300, 0, 100, 100, (max(0, int(random.random() * 256)), max(0, int(random.random() * 256)), max(0, int(random.random() * 256)))),
    ColorRect(400, 0, 100, 100, (0, 0, 255)),
    ColorRect(500, 0, 100, 100, (255, 0, 0)),
    ColorRect(600, 0, 100, 100, (0, 255, 0)),
    ColorRect(700, 0, 100, 100, (0, 255, 255)),
    ColorRect(800, 0, 100, 100, (0, 0, 0), "Eraser")
]
clear = ColorRect(900, 0, 100, 100, (100, 100, 100), "Clear")
penBtn = ColorRect(1100, 0, 100, 50, color, 'Pen')
pens = [ColorRect(1100, 50 + 100 * i, 100, 100, (50, 50, 50), str(penSize)) for i, penSize in enumerate(range(5, 25, 5))]
boardBtn = ColorRect(50, 0, 100, 100, (255, 255, 0), 'Board')
whiteBoard = ColorRect(50, 120, 1020, 580, (255, 255, 255), alpha=0.6)

coolingCounter = 0
hideBoard = True
hideColors = True
hidePenSizes = True

while True:
    if coolingCounter > 0:
        coolingCounter -= 1

    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame")
        break
    frame = cv2.resize(frame, (1280, 720))
    frame = cv2.flip(frame, 1)

    detector.findHands(frame, draw=False)
    positions = detector.getPosition(frame, draw=False)
    upFingers = detector.getUpFingers(frame)

    if upFingers and len(positions) > 8:
        x, y = positions[8]
        if upFingers[1] and not whiteBoard.isOver(x, y):
            px, py = 0, 0
            if not hidePenSizes:
                for pen in pens:
                    if pen.isOver(x, y):
                        brushSize = int(pen.text)
                        pen.alpha = 0
                    else:
                        pen.alpha = 0.5
            if not hideColors:
                for cb in colors:
                    if cb.isOver(x, y):
                        color = cb.color
                        cb.alpha = 0
                    else:
                        cb.alpha = 0.5
                if clear.isOver(x, y):
                    clear.alpha = 0
                    canvas = np.zeros((720, 1280, 3), np.uint8)
                else:
                    clear.alpha = 0.5
            if colorsBtn.isOver(x, y) and coolingCounter == 0:
                coolingCounter = 10
                hideColors = not hideColors
                colorsBtn.text = 'Colors' if hideColors else 'Hide'
                colorsBtn.alpha = 0
            else:
                colorsBtn.alpha = 0.5
            if penBtn.isOver(x, y) and coolingCounter == 0:
                coolingCounter = 10
                hidePenSizes = not hidePenSizes
                penBtn.text = 'Pen' if hidePenSizes else 'Hide'
                penBtn.alpha = 0
            else:
                penBtn.alpha = 0.5
            if boardBtn.isOver(x, y) and coolingCounter == 0:
                coolingCounter = 10
                hideBoard = not hideBoard
                boardBtn.text = 'Board' if hideBoard else 'Hide'
                boardBtn.alpha = 0
            else:
                boardBtn.alpha = 0.5
        elif upFingers[1] and not upFingers[2] and not hideBoard:
            if whiteBoard.isOver(x, y):
                cv2.circle(frame, (x, y), brushSize, color, -1)
                if px == 0 and py == 0:
                    px, py = x, y
                size = eraserSize if color == (0, 0, 0) else brushSize
                cv2.line(canvas, (px, py), (x, y), color, size)
                px, py = x, y
        else:
            px, py = 0, 0

    colorsBtn.drawRect(frame)
    cv2.rectangle(frame, (colorsBtn.x, colorsBtn.y), (colorsBtn.x + colorsBtn.w, colorsBtn.y + colorsBtn.h), (255, 255, 255), 2)
    boardBtn.drawRect(frame)
    cv2.rectangle(frame, (boardBtn.x, boardBtn.y), (boardBtn.x + boardBtn.w, boardBtn.y + boardBtn.h), (255, 255, 255), 2)

    if not hideBoard:
        whiteBoard.drawRect(frame)
        canvas_gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        _, img_inv = cv2.threshold(canvas_gray, 20, 255, cv2.THRESH_BINARY_INV)
        img_inv = cv2.cvtColor(img_inv, cv2.COLOR_GRAY2BGR)
        frame = cv2.bitwise_and(frame, img_inv)
        frame = cv2.bitwise_or(frame, canvas)

    if not hideColors:
        for c in colors:
            c.drawRect(frame)
            cv2.rectangle(frame, (c.x, c.y), (c.x + c.w, c.y + c.h), (255, 255, 255), 2)
        clear.drawRect(frame)
        cv2.rectangle(frame, (clear.x, clear.y), (clear.x + clear.w, clear.y + clear.h), (255, 255, 255), 2)

    penBtn.color = color
    penBtn.drawRect(frame)
    cv2.rectangle(frame, (penBtn.x, penBtn.y), (penBtn.x + penBtn.w, penBtn.y + penBtn.h), (255, 255, 255), 2)
    if not hidePenSizes:
        for pen in pens:
            pen.drawRect(frame)
            cv2.rectangle(frame, (pen.x, pen.y), (pen.x + pen.w, pen.y + pen.h), (255, 255, 255), 2)

    cv2.imshow('video', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
