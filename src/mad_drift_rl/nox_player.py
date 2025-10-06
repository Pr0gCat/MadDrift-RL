import subprocess
import logging
import win32api
import win32con
import time
from dataclasses import dataclass

def nox(*args, **kwargs):
    return subprocess.run(["NoxConsole", *args], **kwargs)

@dataclass
class EmulatorInfo:
    id: int
    name: str
    top_window_handle: int
    toolbar_window_handle: int
    bind_window_handle: int
    pid: int
    
    @property
    def is_running(self):
        return self.pid > 0

class NoxPlayer:
    @staticmethod
    def check_availability():
        try:
            nox(check=True, capture_output=True)
            return True
        except:
            logging.error("NoxConsole not found. Please ensure it is installed and accessible in your PATH.")
        return False

    @staticmethod
    def quit_all_instances():
        try:
            logging.info("Quitting all Nox instances...")
            nox("quitall", check=True)
        except Exception as e:
            logging.error(f"Failed to quit all Nox instances: {e}")

    @staticmethod
    def list_emulators() -> list[EmulatorInfo]:
        try:
            emulators: list[EmulatorInfo] = []
            for entry in nox("list", capture_output=True, text=True, check=True).stdout.splitlines():
                info = entry.strip().split(',')
                assert len(info) == 7, f"Unexpected Nox list output: {info}"
                    
                emulators.append(
                    EmulatorInfo(
                        id=int(info[0]),
                        name=info[2],
                        top_window_handle=int(info[3], 16),
                        toolbar_window_handle=int(info[4], 16),
                        bind_window_handle=int(info[5], 16),
                        pid=int(info[6])  # PID is at index 6, not 5
                    )
                )
            return emulators
        except Exception as e:
            logging.error(f"Failed to list Nox instances: {e}")
        return []
    
    @staticmethod
    def launch(emulator_info: EmulatorInfo):
        try:
            logging.info(f"Launching Nox instance {emulator_info.name}({emulator_info.id})...")
            nox("launch", f"-index:{emulator_info.id}", check=True)
        except Exception as e:
            logging.error(f"Failed to launch Nox instance {emulator_info.id}: {e}")
            
        # wait for adb to be available
        import time
        while True:
            try:
                NoxPlayer.adb(emulator_info.id, "devices")
                break
            except Exception as e:
                logging.info("Waiting for adb to be available...")
                time.sleep(1)
    
    @staticmethod
    def adb(*args):
        # TODO: matching device with emulator id
        return subprocess.run(["adb", *args], check=True)
    
    @staticmethod
    def click(emulator_info: EmulatorInfo, x: int, y: int):
        # TODO: use pywin32 to click directly on the window
        try:
            NoxPlayer.adb("shell", "input", "tap", str(x), str(y))
        except Exception as e:
            logging.error(f"Failed to click at ({x}, {y}) on Nox instance {emulator_info.id}: {e}")

    @staticmethod
    def tap_and_hold(emulator_info: EmulatorInfo, x: int, y: int, duration_ms: int):
        """Hold touch at position for specified duration using direct win32 input.

        Args:
            emulator_info: The emulator to send the command to
            x: X coordinate to hold (in emulator coordinates)
            y: Y coordinate to hold (in emulator coordinates)
            duration_ms: Duration to hold in milliseconds

        Note: Uses win32api to send mouse events directly to the window, bypassing ADB
        """
        try:
            hwnd = emulator_info.top_window_handle

            # Convert emulator coordinates to window coordinates (accounting for window chrome)
            # Nox has a 2px left border and 32px top border (title bar)
            window_x = x + 2
            window_y = y + 32

            # Create lParam for PostMessage (x in low word, y in high word)
            lParam = win32api.MAKELONG(window_x, window_y)

            # Send mouse down event
            win32api.PostMessage(hwnd, win32con.WM_LBUTTONDOWN, win32con.MK_LBUTTON, lParam)

            # Hold for duration
            time.sleep(duration_ms / 1000.0)

            # Send mouse up event
            win32api.PostMessage(hwnd, win32con.WM_LBUTTONUP, 0, lParam)

        except Exception as e:
            logging.error(f"Failed to hold at ({x}, {y}) for {duration_ms}ms on Nox instance {emulator_info.id}: {e}")
            