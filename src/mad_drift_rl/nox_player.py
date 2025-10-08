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

    def get_window_borders(self) -> tuple[int, int]:
        """Get the actual window border offsets (left, top) for this emulator.

        Returns:
            tuple: (left_border, top_border) in pixels
        """
        import win32gui
        window_rect = win32gui.GetWindowRect(self.top_window_handle)
        screen_point = win32gui.ClientToScreen(self.top_window_handle, (0, 0))

        left_border = screen_point[0] - window_rect[0]
        top_border = screen_point[1] - window_rect[1]

        return (left_border, top_border)

    def get_android_resolution(self) -> tuple[int, int]:
        """Get the internal Android resolution via ADB.

        Returns:
            tuple: (width, height) of Android display in pixels
        """
        import subprocess
        try:
            result = subprocess.run(
                ["adb", "shell", "wm", "size"],
                capture_output=True,
                text=True,
                check=True,
                timeout=2
            )
            # Parse output like "Physical size: 540x960"
            for line in result.stdout.split('\n'):
                if 'Physical size:' in line:
                    size_str = line.split(':')[1].strip()
                    width, height = map(int, size_str.split('x'))
                    return (width, height)
        except Exception as e:
            logging.warning(f"Failed to get Android resolution via ADB: {e}")

        # Fallback to common Nox default
        return (540, 960)

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
        """Click at position using win32 PostMessage (fast, non-blocking).

        Args:
            emulator_info: The emulator to send the command to
            x: X coordinate to click (in screenshot coordinates)
            y: Y coordinate to click (in screenshot coordinates)
        """
        try:
            hwnd = emulator_info.top_window_handle

            # Get window client size to account for DPI scaling
            import win32gui
            rect = win32gui.GetClientRect(hwnd)
            window_width = rect[2] - rect[0]
            window_height = rect[3] - rect[1]

            # Get Android resolution
            android_width, android_height = emulator_info.get_android_resolution()

            # Scale from screenshot coordinates (window size) to Android coordinates
            # This handles Windows DPI scaling (e.g., 150% = 448x795 window showing 540x960 content)
            android_x = int(x * android_width / window_width)
            android_y = int(y * android_height / window_height)

            logging.debug(f"Click: screenshot ({x}, {y}) -> Android ({android_x}, {android_y})")

            # Create lParam for PostMessage
            lParam = win32api.MAKELONG(android_x, android_y)

            # Send mouse events - fast and non-blocking
            win32api.PostMessage(hwnd, win32con.WM_LBUTTONDOWN, win32con.MK_LBUTTON, lParam)
            win32api.PostMessage(hwnd, win32con.WM_LBUTTONUP, 0, lParam)

        except Exception as e:
            logging.error(f"Failed to click at ({x}, {y}) on Nox instance {emulator_info.id}: {e}")

    @staticmethod
    def tap_and_hold(emulator_info: EmulatorInfo, x: int, y: int, duration_ms: int):
        """Hold touch at position for specified duration using win32 PostMessage.

        Args:
            emulator_info: The emulator to send the command to
            x: X coordinate to hold (in screenshot coordinates)
            y: Y coordinate to hold (in screenshot coordinates)
            duration_ms: Duration to hold in milliseconds
        """
        try:
            hwnd = emulator_info.top_window_handle

            # Get window client size to account for DPI scaling
            import win32gui
            rect = win32gui.GetClientRect(hwnd)
            window_width = rect[2] - rect[0]
            window_height = rect[3] - rect[1]

            # Get Android resolution
            android_width, android_height = emulator_info.get_android_resolution()

            # Scale from screenshot coordinates (window size) to Android coordinates
            # This handles Windows DPI scaling (e.g., 150% = 448x795 window showing 540x960 content)
            android_x = int(x * android_width / window_width)
            android_y = int(y * android_height / window_height)

            # Create lParam for PostMessage
            lParam = win32api.MAKELONG(android_x, android_y)

            # Send mouse down
            win32api.PostMessage(hwnd, win32con.WM_LBUTTONDOWN, win32con.MK_LBUTTON, lParam)

            # Hold for duration
            time.sleep(duration_ms / 1000.0)

            # Send mouse up
            win32api.PostMessage(hwnd, win32con.WM_LBUTTONUP, 0, lParam)

        except Exception as e:
            logging.error(f"Failed to hold at ({x}, {y}) for {duration_ms}ms on Nox instance {emulator_info.id}: {e}")
            