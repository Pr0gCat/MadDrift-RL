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
from mad_drift_rl.positions import SCORE_REGION_PCT, CAR_SAMPLE_POINT_PCT, FLOOR_SAMPLE_REGIONS_PCT
from mad_drift_rl.vision_utils import locate, ocr, sample_color, normalize_observation, get_closest_floor_color

class ActionSpace(IntEnum):
    NoOp = 0
    Left = 1
    Right = 2

class MadDriftEnv:
    def __init__(self, emulator: EmulatorInfo):
        self.emulator = emulator

        # Get Android resolution (the actual game resolution)
        self.android_width, self.android_height = emulator.get_android_resolution()

        # Get window size (DPI-scaled version of Android resolution)
        rect = win32gui.GetClientRect(self.emulator.top_window_handle)
        self.width = rect[2] - rect[0] -2
        self.height = rect[3] - rect[1] -32

        # Get window border offsets (varies by DPI/theme)
        self.border_left, self.border_top = emulator.get_window_borders()

        logging.info(f"Window client size: {self.width}x{self.height}")
        logging.info(f"Android resolution: {self.android_width}x{self.android_height}")
        logging.info(f"Window borders: left={self.border_left}px, top={self.border_top}px")

        self.current_step = 0
        self.max_steps = 10000  # Timeout after ~25 minutes at 150ms per step

        # Cache window device contexts - use client area only (no borders/titlebar)
        self.hwndDC = win32gui.GetDC(self.emulator.top_window_handle)  # Use GetDC instead of GetWindowDC
        self.mfcDC = win32ui.CreateDCFromHandle(self.hwndDC)
        self.saveDC = self.mfcDC.CreateCompatibleDC()
        self.saveBitMap = win32ui.CreateBitmap()
        self.saveBitMap.CreateCompatibleBitmap(self.mfcDC, self.width, self.height)
        self.saveDC.SelectObject(self.saveBitMap)

        # Load button templates
        replay_template_original = cv2.imread("assets/templates/replay_btn.jpg")
        skip_template_original = cv2.imread("assets/templates/skip_btn.jpg")

        # Scale templates to match current window DPI
        # Templates were captured at 540x960 (100% DPI), scale them to current window size
        template_scale = self.width / 540.0  # DPI scale factor

        if template_scale != 1.0:
            # Scale templates to match current DPI
            replay_w = int(replay_template_original.shape[1] * template_scale)
            replay_h = int(replay_template_original.shape[0] * template_scale)
            self.replay_btn_template = cv2.resize(replay_template_original, (replay_w, replay_h), interpolation=cv2.INTER_LINEAR)

            skip_w = int(skip_template_original.shape[1] * template_scale)
            skip_h = int(skip_template_original.shape[0] * template_scale)
            self.skip_btn_template = cv2.resize(skip_template_original, (skip_w, skip_h), interpolation=cv2.INTER_LINEAR)

            logging.info(f"Scaled templates by {template_scale:.3f}x to match window DPI")
        else:
            self.replay_btn_template = replay_template_original
            self.skip_btn_template = skip_template_original

        # Theme color storage for normalization
        self.floor_color = None
        self.car_color = None

        # Button position and color caching for fast detection
        self.replay_btn_pos = None
        self.replay_btn_color = None
        self.skip_btn_pos = None
        self.skip_btn_color = None
        self.calibrated = False
    
    def __del__(self):
        # Clean up resources
        try:
            win32gui.DeleteObject(self.saveBitMap.GetHandle())
            self.saveDC.DeleteDC()
            self.mfcDC.DeleteDC()
            win32gui.ReleaseDC(self.emulator.top_window_handle, self.hwndDC)  # ReleaseDC works for both GetDC and GetWindowDC
        except:
            pass
        
    def get_screenshot(self) -> np.ndarray:
        # Capture client area only (no borders/titlebar)
        # The window shows Android 540x960 content scaled by Windows DPI
        self.saveDC.BitBlt(
            (0, 0),
            (self.width, self.height),
            self.mfcDC,
            (2, 32),  # No offset needed since we're using client DC
            win32con.SRCCOPY
        )

        # Convert the bitmap to a PIL Image
        bmpinfo = self.saveBitMap.GetInfo()
        bmpstr = self.saveBitMap.GetBitmapBits(True)
        img = Image.frombuffer('RGB', (bmpinfo['bmWidth'], bmpinfo['bmHeight']), bmpstr, 'raw', 'BGRX', 0, 1)

        return np.array(img)

    def _scale_region(self, region_pct: tuple[float, float, float, float]) -> tuple[int, int, int, int]:
        """Convert percentage-based region to pixel coordinates for actual window size.

        Args:
            region_pct: (x%, y%, width%, height%) as floats 0.0-1.0

        Returns:
            (x, y, width, height) in pixels
        """
        x_pct, y_pct, w_pct, h_pct = region_pct
        return (
            int(x_pct * self.width),
            int(y_pct * self.height),
            int(w_pct * self.width),
            int(h_pct * self.height)
        )
    
    def click(self, x: int, y: int):
        # Apply offset correction for emulator coordinate system
        NoxPlayer.click(self.emulator, x + 2, y + 32)
    
    def tap_left(self):
        # Use percentage-based positioning (10% from left, 95% from top)
        x = int(self.width * 0.10)
        y = int(self.height * 0.95)
        self.click(x, y)

    def tap_right(self):
        # Use percentage-based positioning (90% from left, 95% from top)
        x = int(self.width * 0.90)
        y = int(self.height * 0.95)
        self.click(x, y)

    def tap_and_hold_left(self, duration: float):
        """Hold left button for specified duration.

        Args:
            duration: Duration to hold in seconds (converted to milliseconds for ADB)
        """
        duration_ms = int(duration * 1000)
        # Use percentage-based positioning (10% from left, 95% from top)
        x = int(self.width * 0.10)
        y = int(self.height * 0.95)
        NoxPlayer.tap_and_hold(self.emulator, x, y, duration_ms)

    def tap_and_hold_right(self, duration: float):
        """Hold right button for specified duration.

        Args:
            duration: Duration to hold in seconds (converted to milliseconds for ADB)
        """
        duration_ms = int(duration * 1000)
        # Use percentage-based positioning (90% from left, 95% from top)
        x = int(self.width * 0.90)
        y = int(self.height * 0.95)
        NoxPlayer.tap_and_hold(self.emulator, x, y, duration_ms)
    
    def find_skip_button(self, screenshot: np.ndarray) -> tuple[Any, Any, float] | None:
        # Convert RGB screenshot to BGR for template matching (templates are loaded with cv2.imread in BGR)
        screenshot_bgr = cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)
        # Use lower threshold for skip button - it may vary more in appearance
        result = locate(screenshot_bgr, self.skip_btn_template, threshold=0.7, multi_scale=True)
        if result:
            # Template matching returns center of template, adjust to click lower on the button
            x, y, confidence = result
            # Add small Y offset to click center/lower part of button (more reliable)
            adjusted_y = y + int(10 * (self.height / 795.0))  # Scale offset based on window size
            logging.info(f"Skip button found at ({x}, {y}), adjusted to ({x}, {adjusted_y}) with confidence {confidence:.3f}")
            return (x, adjusted_y, confidence)
        else:
            logging.debug("Skip button not found")
        return result

    def is_game_over(self, screenshot: np.ndarray) -> tuple[Any, Any, float] | None:
        # Convert RGB screenshot to BGR for template matching (templates are loaded with cv2.imread in BGR)
        screenshot_bgr = cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)
        result = locate(screenshot_bgr, self.replay_btn_template, threshold=0.8, multi_scale=True)
        if result:
            logging.info(f"Replay button found at ({result[0]}, {result[1]}) with confidence {result[2]:.3f}")
        else:
            logging.debug("Replay button not found")
        return result
    
    def make_observation(self, screenshot: np.ndarray) -> np.ndarray:
        # Crop RGB game view first (9.4% from top, extending 78.5% of height)
        top = int(0.094 * self.height)
        bottom = int(0.786 * self.height)  # 0.094 + 0.692
        game_view_rgb = screenshot[top:bottom, :]
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
    
    def calibrate_buttons(self):
        """Calibrate button positions and colors for fast detection.

        Should be called when game is at game over screen.
        Sequence:
        1. Sample replay button color (game over screen)
        2. Click replay to start game
        3. Wait for skip button to appear
        4. Sample skip button color
        5. Click skip
        """
        logging.info("Calibrating button positions and colors...")

        # Step 1: Sample replay button
        screenshot = self.get_screenshot()
        
        screenshot_bgr = cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)
        replay_result = locate(screenshot_bgr, self.replay_btn_template, threshold=0.8, multi_scale=True)

        if not replay_result:
            raise RuntimeError("Failed to locate replay button for calibration")

        self.replay_btn_pos = (replay_result[0], replay_result[1])
        self.replay_btn_color = screenshot[self.replay_btn_pos[1], self.replay_btn_pos[0]]
        logging.info(f"Replay button: pos={self.replay_btn_pos}, color={self.replay_btn_color}")

        # Step 2: Click replay to start game
        self.click(replay_result[0], replay_result[1])
        time.sleep(0.5)

        # Step 3: Wait for skip button to appear (timeout 10 seconds)
        skip_found = False
        start_time = time.time()
        while time.time() - start_time < 10.0:
            screenshot = self.get_screenshot()
            screenshot_bgr = cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)
            skip_result = locate(screenshot_bgr, self.skip_btn_template, threshold=0.7, multi_scale=True)

            if skip_result:
                # Adjust Y position for better clicking (same as find_skip_button)
                adjusted_y = skip_result[1] + int(10 * (self.height / 795.0))
                self.skip_btn_pos = (skip_result[0], adjusted_y)
                self.skip_btn_color = screenshot[adjusted_y, skip_result[0]]
                logging.info(f"Skip button: pos={self.skip_btn_pos}, color={self.skip_btn_color}")
                skip_found = True
                break

            cv2.waitKey(1)  # Process GUI events to prevent freezing
            time.sleep(0.1)

        if not skip_found:
            raise RuntimeError("Failed to locate skip button for calibration")

        # Step 4: Click skip
        self.click(self.skip_btn_pos[0], self.skip_btn_pos[1])
        time.sleep(0.1)

        # Mark as calibrated now so we can use fast checks
        self.calibrated = True

        # Wait for game over screen again
        while True:
            logging.info("Waiting for game over screen to reappear...")
            screenshot = self.get_screenshot()
            if self._check_game_over_fast(screenshot):
                break
            time.sleep(0.1)

        logging.info("Button calibration complete!")

    def _check_game_over_fast(self, screenshot: np.ndarray) -> bool:
        """Fast game over check using cached position and color, with template matching verification."""
        if not self.calibrated:
            return False

        # Sample pixel at cached position
        pixel_color = screenshot[self.replay_btn_pos[1], self.replay_btn_pos[0]]

        # Check if color matches (with strict tolerance)
        color_diff = np.abs(pixel_color.astype(np.int32) - self.replay_btn_color.astype(np.int32))
        color_match = np.all(color_diff <= 5)  # Allow ±5 RGB difference (tighter tolerance)

        # If color matches, double-check with template matching to avoid false positives
        if color_match:
            screenshot_bgr = cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)
            result = locate(screenshot_bgr, self.replay_btn_template, threshold=0.8, multi_scale=True)
            return result is not None

        return False

    def _check_skip_button_fast(self, screenshot: np.ndarray) -> bool:
        """Fast skip button check using cached position and color."""
        if not self.calibrated:
            return False

        # Sample pixel at cached position
        pixel_color = screenshot[self.skip_btn_pos[1], self.skip_btn_pos[0]]

        # Check if color matches (with strict tolerance)
        color_diff = np.abs(pixel_color.astype(np.int32) - self.skip_btn_color.astype(np.int32))
        return np.all(color_diff <= 5)  # Allow ±5 RGB difference (tighter tolerance)

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
        # Wait for and click replay button (appears after skip)
        while True:
            screenshot = self.get_screenshot()

            if self._check_game_over_fast(screenshot):
                self.click(self.replay_btn_pos[0], self.replay_btn_pos[1])
                # Wait for game to start and car to appear for color sampling
                time.sleep(0.5)
                self.current_step = 0

                # Sample theme colors for normalization (from RGB screenshot)
                screenshot = self.get_screenshot()

                # Sample car color from bottom center (RGB) - sample single point
                car_x = int(CAR_SAMPLE_POINT_PCT[0] * self.width)
                car_y = int(CAR_SAMPLE_POINT_PCT[1] * self.height)
                self.car_color = screenshot[car_y, car_x]

                # Sample floor color from multiple clean regions (average for robustness)
                floor_samples = [sample_color(screenshot, self._scale_region(region)) for region in FLOOR_SAMPLE_REGIONS_PCT]
                sampled_floor_color = np.mean(floor_samples, axis=0).astype(np.uint8)

                # Get the closest matching hardcoded floor color
                self.floor_color = get_closest_floor_color(sampled_floor_color)

                logging.info(f"Theme colors sampled - Sampled: {sampled_floor_color}, Matched: {self.floor_color}, Car: {self.car_color}")

                # Return first observation with normalization
                return self.make_observation(screenshot)
            cv2.waitKey(1)  # Process GUI events to prevent freezing
            time.sleep(0.1)

                
    def step(self, action: ActionSpace, interval: float = 0.15) -> tuple[float, np.ndarray | None, bool, dict]:
        """
        Returns:
            reward: float - the reward for this step
            observation: np.ndarray | None - the next observation (None if done)
            done: bool - whether the episode is finished
            info: dict - additional information (e.g., final score)
        """
        import time as time_module
        step_start = time_module.time()

        t1 = time_module.time()
        screenshot = self.get_screenshot()
        logging.debug(f"  get_screenshot: {(time_module.time() - t1)*1000:.2f}ms")
        self.current_step += 1

        # Check for skip button first (appears before replay button after crash)
        t2 = time_module.time()
        if self._check_skip_button_fast(screenshot):
            logging.debug(f"  check_skip_button: {(time_module.time() - t2)*1000:.2f}ms (found)")
            time.sleep(0.5)  # Wait before clicking skip
            self.click(self.skip_btn_pos[0], self.skip_btn_pos[1])
            time.sleep(0.1)
            # Update screenshot after clicking skip
            screenshot = self.get_screenshot()
            while not self._check_game_over_fast(screenshot):
                time.sleep(0.1)
                screenshot = self.get_screenshot()
        else:
            logging.debug(f"  check_skip_button: {(time_module.time() - t2)*1000:.2f}ms (not found)")

        # Check if game over (replay button visible)
        # IMPORTANT: Return early with None observation to prevent game over UI from entering training data
        t3 = time_module.time()
        if self._check_game_over_fast(screenshot):
            logging.debug(f"  check_game_over: {(time_module.time() - t3)*1000:.2f}ms (game over)")
            # take score (use percentage-based region)
            scaled_score_region = self._scale_region(SCORE_REGION_PCT)
            score_region = screenshot[
                scaled_score_region[1]:scaled_score_region[1]+scaled_score_region[3],
                scaled_score_region[0]:scaled_score_region[0]+scaled_score_region[2]
            ]
            # cv2.imshow("Score Region", score_region)
            score_text = ocr(score_region)
            logging.info(f"OCR Score Text: '{score_text}'")

            try:
                final_score = int(score_text)
                return final_score, None, True, {"final_score": final_score, "timeout": False, "ocr_failed": False}
            except (ValueError, TypeError):
                logging.warning(f"Failed to parse OCR score text: '{score_text}'")
                return 0, None, True, {"final_score": 0, "timeout": False, "ocr_failed": True}
        else:
            logging.debug(f"  is_game_over: {(time_module.time() - t3)*1000:.2f}ms (not game over)")

        # Check timeout
        if self.current_step >= self.max_steps:
            return 0.0, None, True, {"final_score": 0, "timeout": True}

        # Execute action with hold duration
        # For Left/Right: hold the button for 'interval' seconds (blocks during execution)
        # For NoOp: just wait for 'interval' seconds
        t4 = time_module.time()
        match action:
            case ActionSpace.Left:
                self.tap_and_hold_left(interval)  # Blocks for interval seconds
            case ActionSpace.Right:
                self.tap_and_hold_right(interval)  # Blocks for interval seconds
            case ActionSpace.NoOp:
                time.sleep(interval)  # Wait for interval seconds
        logging.debug(f"  action_execution: {(time_module.time() - t4)*1000:.2f}ms")

        # Get fresh screenshot AFTER action execution
        t5 = time_module.time()
        screenshot_after_action = self.get_screenshot()
        logging.debug(f"  get_screenshot_after: {(time_module.time() - t5)*1000:.2f}ms")

        # CRITICAL: Check for skip/game-over on the NEW screenshot (car may have crashed during action)
        # Must check BEFORE creating observation to prevent game over UI from contaminating training data

        # Check for skip button (appears after crash)
        t6 = time_module.time()
        if self._check_skip_button_fast(screenshot_after_action):
            logging.debug(f"  check_skip_button_after: {(time_module.time() - t6)*1000:.2f}ms (found)")
            time.sleep(0.5)  # Wait before clicking skip
            self.click(self.skip_btn_pos[0], self.skip_btn_pos[1])
            time.sleep(0.1)
            # Wait for replay button to appear
            while True:
                screenshot_after_action = self.get_screenshot()
                if self._check_game_over_fast(screenshot_after_action):
                    break
                time.sleep(0.1)
        else:
            logging.debug(f"  check_skip_button_after: {(time_module.time() - t6)*1000:.2f}ms (not found)")

        # Check if game over (replay button visible)
        t7 = time_module.time()
        if self._check_game_over_fast(screenshot_after_action):
            logging.debug(f"  check_game_over_after: {(time_module.time() - t7)*1000:.2f}ms (game over)")
            # take score (use percentage-based region)
            scaled_score_region = self._scale_region(SCORE_REGION_PCT)
            score_region = screenshot_after_action[
                scaled_score_region[1]:scaled_score_region[1]+scaled_score_region[3],
                scaled_score_region[0]:scaled_score_region[0]+scaled_score_region[2]
            ]
            score_text = ocr(score_region)
            logging.info(f"OCR Score Text: '{score_text}'")

            try:
                final_score = int(score_text)
                return final_score, None, True, {"final_score": final_score, "timeout": False, "ocr_failed": False}
            except (ValueError, TypeError):
                logging.warning(f"Failed to parse OCR score text: '{score_text}'")
                return 0, None, True, {"final_score": 0, "timeout": False, "ocr_failed": True}
        else:
            logging.debug(f"  is_game_over_after: {(time_module.time() - t7)*1000:.2f}ms (not game over)")

        # Survival reward (small positive reward for each step survived)
        t8 = time_module.time()
        reward = 0.01
        observation = self.make_observation(screenshot_after_action)
        logging.debug(f"  make_observation: {(time_module.time() - t8)*1000:.2f}ms")

        logging.debug(f"  TOTAL step time: {(time_module.time() - step_start)*1000:.2f}ms")
        return reward, observation, False, {}