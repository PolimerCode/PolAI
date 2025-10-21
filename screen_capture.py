import time
import cv2
import numpy as np
import mss

def get_capture_monitor(left, top, right, bottom, title_bar=30, border_lr=8, border_bottom=8):
    capture_left = left + border_lr
    capture_top = top + title_bar
    capture_width = (right - left) - 2 * border_lr
    capture_height = (bottom - top) - title_bar - border_bottom
    return {
        "left": capture_left,
        "top": capture_top,
        "width": capture_width,
        "height": capture_height
    }

def show_window(monitor):
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
    # user window config
    monitor = get_capture_monitor(525, 269, 1395, 788)
    print("Capture area:", monitor)
    show_window(monitor)
