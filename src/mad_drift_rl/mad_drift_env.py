import win32gui
import win32ui
import win32con
import numpy as np
from PIL import Image
import time

from mad_drift_rl.nox_player import NoxPlayer, EmulatorInfo

EMULATOR_PREFIX = "MadDriftEnv"

class MadDriftEnv:
    def __init__(self, emulator: EmulatorInfo, fps: int = 30):
        self.emulator = emulator
        self.hwnd = emulator.top_window_handle
        self.width = 540
        self.height = 960
        self.fps = fps
        self.frame_interval = 1.0 / fps
        self._last_screenshot = None
        self._last_capture_time = 0

        # Cache window device contexts
        self.hwndDC = win32gui.GetWindowDC(self.hwnd)
        self.mfcDC = win32ui.CreateDCFromHandle(self.hwndDC)
        self.saveDC = self.mfcDC.CreateCompatibleDC()
        self.saveBitMap = win32ui.CreateBitmap()
        self.saveBitMap.CreateCompatibleBitmap(self.mfcDC, self.width, self.height)
        self.saveDC.SelectObject(self.saveBitMap)
    
    def get_screenshot(self) -> tuple[bool, np.ndarray]:
        current_time = time.time()

        # Return cached screenshot if within frame interval
        if self._last_screenshot is not None and (current_time - self._last_capture_time) < self.frame_interval:
            return False, self._last_screenshot

        # Capture new screenshot using cached device contexts
        # Copy the window's content into the bitmap
        self.saveDC.BitBlt((0, 0), (self.width, self.height), self.mfcDC, (2, 32), win32con.SRCCOPY)

        # Convert the bitmap to a PIL Image
        bmpinfo = self.saveBitMap.GetInfo()
        bmpstr = self.saveBitMap.GetBitmapBits(True)
        img = Image.frombuffer('RGB', (bmpinfo['bmWidth'], bmpinfo['bmHeight']), bmpstr, 'raw', 'BGRX', 0, 1)

        # Cache the screenshot
        self._last_screenshot = np.array(img)
        self._last_capture_time = current_time

        return True, self._last_screenshot
    
    

    def __del__(self):
        # Clean up resources
        try:
            win32gui.DeleteObject(self.saveBitMap.GetHandle())
            self.saveDC.DeleteDC()
            self.mfcDC.DeleteDC()
            win32gui.ReleaseDC(self.hwnd, self.hwndDC)
        except:
            pass
    
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

    # Display realtime screenshots
    import cv2
    while True:
        _, screenshot = env.get_screenshot()
        sc = cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)
        cv2.imshow("MadDrift - Realtime", sc)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()