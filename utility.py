import cv2
import time
import numpy as np


class TextHelper:
    def __init__(self) -> None:
        self.bg_color = (0, 0, 0)
        self.color = (255, 255, 255)
        self.text_type = cv2.FONT_HERSHEY_SIMPLEX
        self.line_type = cv2.LINE_AA

    def putText(self, frame, text, coords):
        cv2.putText(frame, text, coords, self.text_type,
                    0.5, self.bg_color, 3, self.line_type)
        cv2.putText(frame, text, coords, self.text_type,
                    0.5, self.color, 1, self.line_type)

    def rectangle(self, frame, p1, p2):
        cv2.rectangle(frame, p1, p2, self.bg_color, 3)
        cv2.rectangle(frame, p1, p2, self.color, 1)


class FPSHandler:
    def __init__(self):
        self.timestamp = time.time() + 1
        self.start = time.time()
        self.frame_cnt = 0

    def next_iter(self):
        self.timestamp = time.time()
        self.frame_cnt += 1

    def fps(self):
        return self.frame_cnt / (self.timestamp - self.start)


def rescale(frame, scale):
    w = int(frame.shape[1]*scale)
    h = int(frame.shape[0]*scale)
    return cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)


def recMask(x, y, w, h, imgH, imgW):
    mask = np.zeros((imgH, imgW), dtype=np.uint8)
    x1, y1 = int(x-w/2), int(y-h/2)
    x2, y2 = int(x+w/2), int(y+h/2)
    cv2.rectangle(mask, (x1, y1),
                  (x2, y2), 255, -1)

    return mask, x1, y1, x2, y2
