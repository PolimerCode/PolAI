import win32gui

def get_minecraft_rect():
    def enumHandler(hwnd, result):
        if win32gui.IsWindowVisible(hwnd):
            name = win32gui.GetWindowText(hwnd)
            if "Minecraft" in name:
                rect = win32gui.GetWindowRect(hwnd)
                result.append(rect)
    result = []
    win32gui.EnumWindows(enumHandler, result)
    return result[0] if result else None

print(get_minecraft_rect())
