# Mad Drift RL Agent (Halted)

> Mad Drift suffers from memory leak issues on emulators after prolonged use, leading to instability/crashes during training.

## Design

An RL agent that plays Mad Drift, an Android racing game, using image recognition and ADB commands. It makes an action every 150ms when the game is running at 30FPS.

The agent uses PPO (Proximal Policy Optimization) with a CNN based vision encoder + Recurrent Transformer Encoder to learn the optimal policy for controlling the car in the game. The state space consists of the game screen, which is processed using image recognition to extract relevant features. The action space consists of discrete actions such as steering left or right and do nothing.


## Setup

1. Install Tesseract
    ```ps
    > winget install UB-Mannheim.TesseractOCR
    ```
2. Add Tesseract to PATH environment variable

    ```C:\Program Files\Tesseract-OCR```
3. Install Python packages

    ```ps
    > uv sync
    ```
    
---

### Nox Prep. Instructions

1. Install Nox

2. Add Nox to PATH environment variable

    ```C:\Program Files (x86)\Nox\bin```

3. Add an Android 7+ emulator with multi-instance manager
    > Android 9 (64bit) is recommended
    
    ![](https://github.com/Pr0gCat/MadDrift-RL/blob/main/assets/images/android_version_select.PNG)

4. Configure the emulator

    * Set CPU to 1 core with 2048MB RAM, set resolution to 540x960 in phone mode
        ![](https://github.com/Pr0gCat/MadDrift-RL/blob/main/assets/images/performance_settings.PNG)
    * Set FPS limit to 30
        ![](https://github.com/Pr0gCat/MadDrift-RL/blob/main/assets/images/fps_settings.PNG)
    * Disable networking to avoid ads
        ![](https://github.com/Pr0gCat/MadDrift-RL/blob/main/assets/images/disable_network.PNG)
    * Fixed window size and force window to be vertical
        ![](https://github.com/Pr0gCat/MadDrift-RL/blob/main/assets/images/ui_settings.PNG)
    * Disable notifications
        ![](https://github.com/Pr0gCat/MadDrift-RL/blob/main/assets/images/disable_notifications.PNG)

5. Launch the emulator

6. After the emulator is started, install Mad Drift

7. Enable ADB debugging in the emulator settings by going to Settings > About tablet > Tap "Build number" 7 times to enable Developer options. Then go to Developer options and enable "USB debugging".

---

## Training 

1. Launch Mad Drift in the emulator and set the game to Game Over screen
2. Run the training script

    ```ps
    > uv run src/mad_drift_rl/train.py
    ```
