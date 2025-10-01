import subprocess
import logging
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
                        pid=int(info[5])
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
    def adb(emulator_id: int, cmd: str):
        return nox("adb", f"-index:{emulator_id}", f'-command:"{cmd}"')
    