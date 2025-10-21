from pynput import mouse, keyboard

def on_press(key):
    try:
        if key.char == 'r':
            x, y = mouse_controller.position
            print(x, y)
        elif key.char == 'q':
            return False
    except AttributeError:
        pass

mouse_controller = mouse.Controller()

with keyboard.Listener(on_press=on_press) as listener:
    listener.join()
