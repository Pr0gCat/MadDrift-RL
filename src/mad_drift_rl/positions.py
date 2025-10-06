
# Position of the score region in the game over screen
# Note: We intentionally leave some margin around the score for better OCR accuracy
SCORE_REGION = (290, 300, 100, 80) # x, y, width, height

GAMEOVER_INDICATOR_POS = (255, 500)
PLAYING_INDICATOR_RGB = (232, 226, 179)  # White color when playing

OBSTACLE_REGION = (0, 90, 540, 540)

# Car sampling region (bottom center, sampled after replay button pressed)
CAR_SAMPLE_REGION = (250, 680, 40, 40)  # x, y, width, height

# Floor sampling regions (clean areas without shadows or obstacles)
# Multiple regions are averaged to get robust floor color estimate
FLOOR_SAMPLE_REGIONS = [
    (50, 200, 30, 30),   # Top left clean area
    (460, 200, 30, 30),  # Top right clean area
    (270, 350, 30, 30),  # Center clean area
]
