import os
import time
import cv2
import numpy as np
import mss
from pynput import keyboard, mouse

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

def record_dataset(monitor, save_dir="dataset"):
    os.makedirs(f"{save_dir}/images", exist_ok=True)
    os.makedirs(f"{save_dir}/actions", exist_ok=True)

    keys_pressed = {"w":0, "a":0, "s":0, "d":0, "space":0, "mouse1":0}

    def on_press(key):
        try:
            if key.char in keys_pressed:
                keys_pressed[key.char] = 1
        except AttributeError:
            if key == keyboard.Key.space:
                keys_pressed["space"] = 1

    def on_release(key):
        try:
            if key.char in keys_pressed:
                keys_pressed[key.char] = 0
        except AttributeError:
            if key == keyboard.Key.space:
                keys_pressed["space"] = 0

    def on_click(x, y, button, pressed):
        if button == mouse.Button.left:
            keys_pressed["mouse1"] = 1 if pressed else 0

    kb_listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    ms_listener = mouse.Listener(on_click=on_click)
    kb_listener.start()
    ms_listener.start()

    frame_count = 0
    with mss.mss() as sct:
        print("recording started. q to exit")
        while True:
            img = np.array(sct.grab(monitor))
            frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

            # save frame
            img_filename = f"{save_dir}/images/frame_{frame_count:05d}.png"
            cv2.imwrite(img_filename, frame)

            # save actions
            action_filename = f"{save_dir}/actions/frame_{frame_count:05d}.npy"
            np.save(action_filename, np.array(list(keys_pressed.values()), dtype=np.uint8))

            # show frame
            cv2.imshow("Recording", frame)

            frame_count += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            time.sleep(0.01)

    cv2.destroyAllWindows()
    kb_listener.stop()
    ms_listener.stop()

if __name__ == "__main__":
    monitor = get_capture_monitor(525, 269, 1395, 788)
    record_dataset(monitor)
