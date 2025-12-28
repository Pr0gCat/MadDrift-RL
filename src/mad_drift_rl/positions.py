
# All positions are now stored as percentages of window dimensions (0.0 to 1.0)
# These will be scaled to actual window size at runtime

# Position of the score region in the game over screen (as percentages)
# Note: We intentionally leave some margin around the score for better OCR accuracy
SCORE_REGION_PCT = (0.5, 0.3125, 0.185, 0.083)  # x%, y%, width%, height%

GAMEOVER_INDICATOR_POS_PCT = (0.472, 0.521)  # x%, y%
PLAYING_INDICATOR_RGB = (232, 226, 179)  # White color when playing

OBSTACLE_REGION_PCT = (0.0, 0.094, 1.0, 0.563)  # x%, y%, width%, height%

# Car sampling point (bottom center, sampled after replay button pressed)
CAR_SAMPLE_POINT_PCT = (0.5, 0.72)  # x%, y%

# Floor sampling regions (clean areas without shadows or obstacles)
# Multiple regions are averaged to get robust floor color estimate
# Sample from bottom areas only - safe from obstacles and game over panel
FLOOR_SAMPLE_REGIONS_PCT = [
    (0.200, 0.650, 0.056, 0.031),  # Bottom left area
    (0.500, 0.650, 0.056, 0.031),  # Bottom center area
    (0.800, 0.650, 0.056, 0.031),  # Bottom right area
]

# Legacy pixel-based values (kept for backward compatibility, based on 540x960)
SCORE_REGION = (290, 300, 100, 80)
CAR_SAMPLE_REGION = (250, 680, 40, 40)
FLOOR_SAMPLE_REGIONS = [
    (50, 200, 30, 30),
    (460, 200, 30, 30),
    (270, 350, 30, 30),
]
