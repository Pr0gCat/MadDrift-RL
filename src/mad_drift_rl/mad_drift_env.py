import logging
from typing import Any
import win32gui
import win32ui
import win32con
import numpy as np
from PIL import Image
import time
import cv2
from enum import IntEnum

from mad_drift_rl.nox_player import NoxPlayer, EmulatorInfo
from mad_drift_rl.positions import SCORE_REGION, CAR_SAMPLE_REGION, FLOOR_SAMPLE_REGIONS
from mad_drift_rl.vision_utils import locate, ocr, sample_color, normalize_observation

class ActionSpace(IntEnum):
    NoOp = 0
    Left = 1
    Right = 2

class MadDriftEnv:
    def __init__(self, emulator: EmulatorInfo):
        self.emulator = emulator
        self.width = 540
        self.height = 960
        self.current_step = 0
        self.max_steps = 10000  # Timeout after ~25 minutes at 150ms per step

        # Cache window device contexts
        self.hwndDC = win32gui.GetWindowDC(self.emulator.top_window_handle)
        self.mfcDC = win32ui.CreateDCFromHandle(self.hwndDC)
        self.saveDC = self.mfcDC.CreateCompatibleDC()
        self.saveBitMap = win32ui.CreateBitmap()
        self.saveBitMap.CreateCompatibleBitmap(self.mfcDC, self.width, self.height)
        self.saveDC.SelectObject(self.saveBitMap)

        self.replay_btn_template = cv2.imread("assets/templates/replay_btn.jpg")
        self.skip_btn_template = cv2.imread("assets/templates/skip_btn.jpg")

        # Theme color storage for normalization
        self.floor_color = None
        self.car_color = None
    
    def __del__(self):
        # Clean up resources
        try:
            win32gui.DeleteObject(self.saveBitMap.GetHandle())
            self.saveDC.DeleteDC()
            self.mfcDC.DeleteDC()
            win32gui.ReleaseDC(self.emulator.top_window_handle, self.hwndDC)
        except:
            pass
        
    def get_screenshot(self) -> np.ndarray:
        # Capture new screenshot using cached device contexts
        # Copy the window's content into the bitmap
        self.saveDC.BitBlt((0, 0), (self.width, self.height), self.mfcDC, (2, 32), win32con.SRCCOPY)

        # Convert the bitmap to a PIL Image
        bmpinfo = self.saveBitMap.GetInfo()
        bmpstr = self.saveBitMap.GetBitmapBits(True)
        img = Image.frombuffer('RGB', (bmpinfo['bmWidth'], bmpinfo['bmHeight']), bmpstr, 'raw', 'BGRX', 0, 1)

        return np.array(img)
    
    def click(self, x: int, y: int):
        NoxPlayer.click(self.emulator, x, y)
    
    def tap_left(self):
        self.click(54, 690)

    def tap_right(self):
        self.click(480, 690)

    def tap_and_hold_left(self, duration: float):
        """Hold left button for specified duration.

        Args:
            duration: Duration to hold in seconds (converted to milliseconds for ADB)
        """
        duration_ms = int(duration * 1000)
        NoxPlayer.tap_and_hold(self.emulator, 54, 690, duration_ms)

    def tap_and_hold_right(self, duration: float):
        """Hold right button for specified duration.

        Args:
            duration: Duration to hold in seconds (converted to milliseconds for ADB)
        """
        duration_ms = int(duration * 1000)
        NoxPlayer.tap_and_hold(self.emulator, 480, 690, duration_ms)
    
    def find_skip_button(self, screenshot: np.ndarray) -> tuple[Any, Any, float] | None:
        # Convert RGB screenshot to BGR for template matching (templates are loaded with cv2.imread in BGR)
        screenshot_bgr = cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)
        return locate(screenshot_bgr, self.skip_btn_template)

    def is_game_over(self, screenshot: np.ndarray) -> tuple[Any, Any, float] | None:
        # Convert RGB screenshot to BGR for template matching (templates are loaded with cv2.imread in BGR)
        screenshot_bgr = cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)
        return locate(screenshot_bgr, self.replay_btn_template)
    
    def make_observation(self, screenshot: np.ndarray) -> np.ndarray:
        # Crop RGB game view first
        game_view_rgb = screenshot[90:640+115, :]
        resized_rgb = cv2.resize(game_view_rgb, (108, 133), interpolation=cv2.INTER_LINEAR)

        # Apply RGB normalization if theme colors have been sampled
        if self.floor_color is not None and self.car_color is not None:
            normalized = normalize_observation(resized_rgb, self.floor_color, self.car_color)
            return normalized
        else:
            # Fallback: convert to grayscale and normalize (shouldn't happen after first reset)
            logging.warning("Theme colors not sampled yet, returning unnormalized observation")
            gray = cv2.cvtColor(resized_rgb, cv2.COLOR_RGB2GRAY)
            return gray.astype(np.float32) / 255.0
    
    def wait_for_game_over(self, timeout: float = 60.0):
        """Wait for the game to reach game over screen"""
        import time as time_module
        start_time = time_module.time()
        while time_module.time() - start_time < timeout:
            screenshot = self.get_screenshot()
            if self.is_game_over(screenshot):
                return True
            time_module.sleep(0.5)
        return False

    def reset(self):
        screenshot = self.get_screenshot()

        # Wait for and click replay button (appears after skip)
        while True:
            screenshot = self.get_screenshot()
            if replay_btn_pos := self.is_game_over(screenshot):
                self.click(replay_btn_pos[0], replay_btn_pos[1])
                # Wait for game to start and car to appear for color sampling
                time.sleep(0.5)
                self.current_step = 0

                # Sample theme colors for normalization (from RGB screenshot)
                screenshot = self.get_screenshot()

                # Sample car color from bottom center (RGB)
                self.car_color = sample_color(screenshot, CAR_SAMPLE_REGION)

                # Sample floor color from multiple clean regions (average for robustness)
                floor_samples = [sample_color(screenshot, region) for region in FLOOR_SAMPLE_REGIONS]
                self.floor_color = np.mean(floor_samples, axis=0).astype(np.uint8)

                logging.info(f"Theme colors sampled - Floor: {self.floor_color}, Car: {self.car_color}")

                # Return first observation with normalization
                return self.make_observation(screenshot)
            time.sleep(0.1)

                
    def step(self, action: ActionSpace, interval: float = 0.15) -> tuple[float, np.ndarray | None, bool, dict]:
        """
        Returns:
            reward: float - the reward for this step
            observation: np.ndarray | None - the next observation (None if done)
            done: bool - whether the episode is finished
            info: dict - additional information (e.g., final score)
        """
        screenshot = self.get_screenshot()
        self.current_step += 1

        # Check for skip button first (appears before replay button after crash)
        if skip_btn_pos := self.find_skip_button(screenshot):
            self.click(skip_btn_pos[0], skip_btn_pos[1])
            time.sleep(0.1)
            # Update screenshot after clicking skip
            screenshot = self.get_screenshot()
            while not self.is_game_over(screenshot):
                time.sleep(0.1)
                screenshot = self.get_screenshot()

        # Check if game over (replay button visible)
        # IMPORTANT: Return early with None observation to prevent game over UI from entering training data
        if self.is_game_over(screenshot):
            # take score
            score_region = screenshot[
                SCORE_REGION[1]:SCORE_REGION[1]+SCORE_REGION[3],
                SCORE_REGION[0]:SCORE_REGION[0]+SCORE_REGION[2]
            ]
            score_text = ocr(score_region)
            logging.info(f"OCR Score Text: '{score_text}'")

            try:
                final_score = int(score_text)
                return final_score, None, True, {"final_score": final_score, "timeout": False, "ocr_failed": False}
            except (ValueError, TypeError):
                logging.warning(f"Failed to parse OCR score text: '{score_text}'")
                return 0, None, True, {"final_score": 0, "timeout": False, "ocr_failed": True}

        # Check timeout
        if self.current_step >= self.max_steps:
            return 0.0, None, True, {"final_score": 0, "timeout": True}

        # Execute action with hold duration
        # For Left/Right: hold the button for 'interval' seconds (blocks during execution)
        # For NoOp: just wait for 'interval' seconds
        match action:
            case ActionSpace.Left:
                self.tap_and_hold_left(interval)  # Blocks for interval seconds
            case ActionSpace.Right:
                self.tap_and_hold_right(interval)  # Blocks for interval seconds
            case ActionSpace.NoOp:
                time.sleep(interval)  # Wait for interval seconds

        # Get fresh screenshot AFTER action execution
        screenshot_after_action = self.get_screenshot()

        # CRITICAL: Check for skip/game-over on the NEW screenshot (car may have crashed during action)
        # Must check BEFORE creating observation to prevent game over UI from contaminating training data

        # Check for skip button (appears after crash)
        if skip_btn_pos := self.find_skip_button(screenshot_after_action):
            self.click(skip_btn_pos[0], skip_btn_pos[1])
            time.sleep(0.1)
            # Wait for replay button to appear
            while True:
                screenshot_after_action = self.get_screenshot()
                if self.is_game_over(screenshot_after_action):
                    break
                time.sleep(0.1)

        # Check if game over (replay button visible)
        if self.is_game_over(screenshot_after_action):
            # take score
            score_region = screenshot_after_action[
                SCORE_REGION[1]:SCORE_REGION[1]+SCORE_REGION[3],
                SCORE_REGION[0]:SCORE_REGION[0]+SCORE_REGION[2]
            ]
            score_text = ocr(score_region)
            logging.info(f"OCR Score Text: '{score_text}'")

            try:
                final_score = int(score_text)
                return final_score, None, True, {"final_score": final_score, "timeout": False, "ocr_failed": False}
            except (ValueError, TypeError):
                logging.warning(f"Failed to parse OCR score text: '{score_text}'")
                return 0, None, True, {"final_score": 0, "timeout": False, "ocr_failed": True}

        # Survival reward (small positive reward for each step survived)
        reward = 0.01
        observation = self.make_observation(screenshot_after_action)
        return reward, observation, False, {}