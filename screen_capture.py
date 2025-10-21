import time
import cv2
import numpy as np
import mss

# coords
left, top, right, bottom = 525, 269, 1395, 788

# примерные размеры рамок Windows
title_bar = 30    # windname
border_lr = 8     # left right border
border_bottom = 8 # bottom border

# game capture area correction
capture_left = left + border_lr
capture_top = top + title_bar
capture_width = (right - left) - 2 * border_lr
capture_height = (bottom - top) - title_bar - border_bottom

monitor = {
    "left": capture_left,
    "top": capture_top,
    "width": capture_width,
    "height": capture_height
}

print("Capture area:", monitor)

def main():
    with mss.mss() as sct:
        while True:
            img = np.array(sct.grab(monitor))
            frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

            cv2.imshow("Minecraft game", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            time.sleep(0.01)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
