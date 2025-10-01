import win32gui
import win32ui
import win32con
import numpy as np
from PIL import Image

from mad_drift_rl.nox_player import NoxPlayer, EmulatorInfo

EMULATOR_PREFIX = "MadDriftEnv"

class MadDriftEnv:
    def __init__(self, emulator: EmulatorInfo):
        self.emulator = emulator
        self.hwnd = emulator.top_window_handle
        self.width = 540
        self.height = 960
    
    def get_screenshot(self) -> np.ndarray:
        # Get the window's device context
        hwndDC = win32gui.GetWindowDC(self.hwnd)
        mfcDC = win32ui.CreateDCFromHandle(hwndDC)
        saveDC = mfcDC.CreateCompatibleDC()

        # Create a bitmap object
        saveBitMap = win32ui.CreateBitmap()
        saveBitMap.CreateCompatibleBitmap(mfcDC, self.width, self.height)
        saveDC.SelectObject(saveBitMap)

        # Copy the window's content into the bitmap
        saveDC.BitBlt((0, 0), (self.width, self.height), mfcDC, (0, 0), win32con.SRCCOPY)

        # Convert the bitmap to a PIL Image
        bmpinfo = saveBitMap.GetInfo()
        bmpstr = saveBitMap.GetBitmapBits(True)
        img = Image.frombuffer('RGB', (bmpinfo['bmWidth'], bmpinfo['bmHeight']), bmpstr, 'raw', 'BGRX', 0, 1)

        # Clean up
        win32gui.DeleteObject(saveBitMap.GetHandle())
        saveDC.DeleteDC()
        mfcDC.DeleteDC()
        win32gui.ReleaseDC(self.hwnd, hwndDC)

        return np.array(img)
    
if __name__ == "__main__":
    if not NoxPlayer.check_availability():
        exit(1)

    emulators = NoxPlayer.list_emulators()
    if not emulators:
        print("No Nox emulators found.")
        exit(1)

    print(emulators[0])
    
    NoxPlayer.launch(emulators[0])
    
    env = MadDriftEnv(emulators[0])
    screenshot = env.get_screenshot()
    img = Image.fromarray(screenshot)
    img.show()