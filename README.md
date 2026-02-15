# game-boy-camera-screenshot-extractor

EXPERIMENTAL: Extracts and cleans a Game Boy Camera image from a screenshot

# Introduction

This repository contains a script that accepts a screenshot or screenshots of a Game Boy Camera screen displaying a Game Boy Camera image and outputs the GameBoy Camera image in 128x112 in four colors. This script is intended to clean Game Boy Camera images retrieved via [Video Capture](https://funtography.online/wiki/Exporting_images_from_the_Game_Boy_Camera#video_capture), admittedly probably the lowest-quality method of transferring Game Boy Camera images. If you do not have the hardware needed to use this script, you will likely find more success at a more affordable price by pursuing other avenues for transferring your Game Boy Camera images. This script aims to accomplish a similar goal with specific hardware that I already own.

To collect the screenshots I used with this script, I used the following tools:

- Game Boy (various)
- Game Boy Camera
- GameCube with Game Boy Player
- Dazzle DVC100 (Low-quality video capture device)
- [OBS Studio](https://obsproject.com/)

DISCLAIMER: This code is almost exclusively AI-generated. It likely contains errors and does not satisfy the need perfectly.

# Setup

```bash
python -m venv .venv
# Unix-based
source .venv/bin/activate
# Windows
./.venv/Scripts/activate
pip install pillow numpy opencv-python
```

# To run

Drag and drop an image onto the script file, or run the script as follows:

```bash
python gbcam_extract.py screenshot1.png screenshot2.png
```

For example, to run with the sample images provided in `sample-screenshots`, run the following:

```bash
python gbcam_extract.py "sample-screenshots/Screenshot 2025-02-22 21-04-01.png" "sample-screenshots/Screenshot 2025-02-22 21-03-26.png" "sample-screenshots/Screenshot 2025-02-22 21-03-34.png" "sample-screenshots/Screenshot 2025-02-22 21-03-43.png" "sample-screenshots/Screenshot 2025-02-24 07-51-02.png"
```
