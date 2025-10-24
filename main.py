import os
import sys
import time
import traceback
import numpy as np
import cv2
import mss
import torch
import torch.nn as nn
import pydirectinput

from screen_capture import get_capture_monitor

print("Python:", sys.version.splitlines()[0])
print("CWD:", os.getcwd())
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 5, stride=2), nn.ReLU(),
            nn.Conv2d(16, 32, 5, stride=2), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 18 * 13, 128), nn.ReLU(),
            nn.Linear(128, 6),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

def load_model(path):
    model = SimpleCNN().to(device)
    if path and os.path.exists(path):
        try:
            model.load_state_dict(torch.load(path, map_location=device))
            print("Model loaded from", path)
        except Exception:
            print("Failed to load model:", path)
            traceback.print_exc()
    else:
        print("Model file not found, using freshly initialized model (no weights). Expected at:", path)
    model.eval()
    return model

ACTION_KEYS = ['w', 's', 'a', 'd', 'space', 'mouse']

def apply_actions(preds, pressed):
    for i, key in enumerate(ACTION_KEYS):
        want = bool(preds[i])
        if key == 'mouse':
            if want and not pressed.get('mouse', False):
                pydirectinput.mouseDown(button='left')
                pressed['mouse'] = True
            elif not want and pressed.get('mouse', False):
                pydirectinput.mouseUp(button='left')
                pressed['mouse'] = False
        else:
            if want and not pressed.get(key, False):
                pydirectinput.keyDown(key)
                pressed[key] = True
            elif not want and pressed.get(key, False):
                pydirectinput.keyUp(key)
                pressed[key] = False

def release_all(pressed):
    for k, v in list(pressed.items()):
        if v:
            if k == 'mouse':
                pydirectinput.mouseUp(button='left')
            else:
                pydirectinput.keyUp(k)
            pressed[k] = False

def preprocess_frame(frame):
    if frame is None:
        return None
    resized = cv2.resize(frame, (120, 160))
    img = resized.astype(np.float32) / 255.0
    tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)
    return tensor

def main():
    model = load_model("minecraft_model.pth")
    monitor = get_capture_monitor(525, 269, 1395, 788)
    print("Capture area:", monitor)
    pressed = {}

    try:
        with mss.mss() as sct:
            # однократная проверка захвата
            try:
                test = np.array(sct.grab(monitor))
                print("Single grab shape:", test.shape)
                if test.size == 0:
                    print("Warning: grabbed image is empty. Проверьте координаты монитора.")
                    print("Monitor dict:", monitor)
                    return
            except Exception:
                print("Failed to grab single frame. Проверьте mss и координаты.")
                traceback.print_exc()
                return

            while True:
                try:
                    sct_img = np.array(sct.grab(monitor))
                    if sct_img is None or sct_img.size == 0:
                        print("Empty frame, skipping iteration")
                        time.sleep(0.5)
                        continue

                    frame = cv2.cvtColor(sct_img, cv2.COLOR_BGRA2BGR)

                    tensor = preprocess_frame(frame)
                    if tensor is None:
                        print("Preprocess failed, got None tensor")
                        break

                    with torch.no_grad():
                        out = model(tensor).cpu().numpy()[0]
                    preds = out > 0.5

                    apply_actions(preds, pressed)

                    labels = ["W","S","A","D","SPACE","MOUSE"]
                    active = [labels[i] for i, x in enumerate(preds) if x]
                    cv2.putText(frame, "Active: " + ",".join(active), (10,30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                    cv2.imshow("Agent view (press q to quit)", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("Quit pressed")
                        break

                    time.sleep(0.01)
                except KeyboardInterrupt:
                    print("Interrupted by user")
                    break
                except Exception:
                    print("Error during main loop:")
                    traceback.print_exc()
                    time.sleep(1)
    finally:
        release_all(pressed)
        cv2.destroyAllWindows()
        print("Cleanup done")

if __name__ == "__main__":
    main()