# game-boy-camera-picture-extractor

EXPERIMENTAL: TypeScript library and offline-ready PWA that extract and clean Game Boy Camera images from pictures

# Try it now

https://tjcouch1.github.io/game-boy-camera-picture-extractor/

You can install the site as a PWA to use it offline - even on your phone.

# Introduction

This repository contains a script and an offline-ready web interface that accept a picture of a Game Boy Camera screen displaying a Game Boy Camera image and output the Game Boy Camera image in 128x112 in four colors. This project is intended to clean Game Boy Camera images retrieved via taking a picture, but it probably would also work with relatively high-quality [Video Capture](https://funtography.online/wiki/Exporting_images_from_the_Game_Boy_Camera#video_capture) (I tried using my [Dazzle DVC100](https://www.pinnaclesys.com/en/products/dazzle/dvd-recorder-hd/), notorious for its low quality video capture, but it did not work). Please note that this is admittedly probably the lowest-quality method of transferring Game Boy Camera images; if you do not have the hardware needed to use this tool, you will likely find more success at a more affordable price by pursuing [other avenues](https://funtography.online/wiki/Exporting_images_from_the_Game_Boy_Camera) for transferring your Game Boy Camera images. This tool aims to accomplish the goal of transferring Game Boy Camera images with specific hardware that I already own.

DISCLAIMER: Though this `README.md` is not AI-generated, the code in this repo is almost exclusively AI-generated. It likely contains errors, bad code quality and standards, and does not satisfy the need perfectly.

(Note that the following [User Instructions](#user-instructions) section is mirrored in the website, so its introduction repeats some information from the previous sections.)

# User Instructions

WARNING: This extraction algorithm is not fully accurate. Please back up your Game Boy Camera images in other ways before deleting them.

You can use this tool on desktop or mobile (fully offline if you install it as a PWA) to transform pictures of a Game Boy Camera image into the actual Game Boy Camera image in 128x112 in four colors.

## Taking pictures that work with this script

To collect the pictures I used with this script, I used the following hardware:

- Game Boy Advance SP AGS-001 (front-lit screen)
- Game Boy Camera (US/Europe)
- Samsung Galaxy S20 Camera

Though I didn't try any other hardware, I expect other hardware like moderate-quality screenshots from video capture devices or pictures of the Game Boy Advance SP AGS-101 screen may work. The picture needs to contain the entire Game Boy screen (not the entire SP screen; just the 160x144 region representing the Game Boy screen) and needs to be clear and bright enough to distinguish the pixels on the screen relatively well. I suspect Japanese Game Boy Cameras may not work very well because [their Frame 02 is different than the Frame 02 in the US/Europe version](https://tcrf.net/Game_Boy_Camera/Regional_Differences#Frames), and I don't see a matching frame in the Japanese version.

I performed the following steps to get the pictures the way they need to be to work with this script:

1. Ideally, find a very dark room (though the script aims to work with various lighting environments).
2. Turn on the Game Boy Advance SP. Make sure the frontlight is turned on. While it boots up, hold Down to use the [0x17 Game Boy Color palette](https://tcrf.net/Notes:Game_Boy_Color_Bootstrap_ROM#Assigned_Palette_Configurations).
3. In the Game Boy Camera game, navigate through the menu options to view the album.
4. Select an image. Once you select it, change the frame to [Frame 02](https://tcrf.net/images/5/50/GBCamera-Frame02-INT.png) (white with black dashes):

   ![Frame 02](https://github.com/tjcouch1/game-boy-camera-screenshot-extractor/blob/main/supporting-materials/Frame%2002.png)

5. Line up the camera with the Game Boy Advance SP screen (approximately; I did this by hand, and it worked fine). Take a picture of the SP screen. Make sure the picture is very well focused and shows the individual pixels on the screen.
6. Crop the image and rotate it so the Game Boy Screen (the 160x144 region that has the frame, not the entire SP screen) is approximately centered with just a little bit of dark screen around it and is relatively straight.

The picture should look like the following:

![Sample picture ready for processing](https://github.com/tjcouch1/game-boy-camera-screenshot-extractor/blob/main/sample-pictures/20260313_213430.jpg)

See [`sample-pictures`](https://github.com/tjcouch1/game-boy-camera-picture-extractor/tree/main/sample-pictures) for more examples of what the pictures need to look like.

## Transforming the picture into a Game Boy Camera image

Upload the picture on [the Game Boy Camera Picture Extractor site](https://tjcouch1.github.io/game-boy-camera-picture-extractor/), and it will automatically begin processing the picture.

After processing, the output Game Boy Camera image should look like the following:

![Sample output picture](https://github.com/tjcouch1/game-boy-camera-screenshot-extractor/blob/main/sample-pictures-out/20260313_213430_gbcam.png)

See [`sample-pictures-out`](https://github.com/tjcouch1/game-boy-camera-picture-extractor/tree/main/sample-pictures-out) for more examples of what the processed images look like.

## Helping to improve the extraction accuracy

The more test cases (input picture + perfect Game Boy Camera image) there are to check the algorithm's accuracy, the higher the chances are that the algorithm will improve with continued development. If you would like to help to improve the accuracy of this script, please [submit a new issue](https://github.com/tjcouch1/game-boy-camera-picture-extractor/issues/new?template=test-images.md) containing the following:

- Unmodified input pictures (photos taken of a Game Boy Camera image - but these should not be cropped or rotated as opposed to the normal input pictures)
- Reference images (128x112 perfect Game Boy Camera output images) - you need [special hardware](https://funtography.online/wiki/Exporting_images_from_the_Game_Boy_Camera) to collect these from an actual Game Boy Camera.

Or feel free to [submit a pull request](https://github.com/tjcouch1/game-boy-camera-picture-extractor/compare) if you improve the algorithm directly!

# Contents

This repository is a pnpm monorepo that contains multiple packages in `/packages`:

- `gbcam-extract` - TypeScript Game Boy Camera image processing pipeline
- `gbcam-extract-web` - Static site hosting the TypeScript Game Boy Camera image processing pipeline for portable and offline use
- `gbcam-extract-py` - original Python Game Boy Camera image processing pipeline (included for historical reasons but will not be updated)

# TypeScript development instructions

TODO: expand

## TypeScript Setup

```bash
pnpm i
```

If you get a message about "ignored build scripts" (`pnpm` seems to be having trouble seeing the build script approval for `esbuild`):

```bash
pnpm rebuild
# or
pnpm approve-builds #and select esbuild
```

## To build the TypeScript packages

```bash
pnpm build
```

## To run TypeScript tests

To run unit tests on TypeScript code:

```bash
pnpm test
```

To run tests to check the pipeline's accuracy against `-output-corrected.png` files in `test-input` (and optionally `test-input-private` if it exists):

```bash
pnpm test:pipeline
```

To run tests as mentioned above and to run the `sample-pictures` and `sample-pictures-full` through the pipeline as well (and optionally `sample-pictures-private` if it exists):

```bash
pnpm test:pipeline:all
```

## To run the extraction pipeline locally in Node

```bash
pnpm extract --dir ../../sample-pictures --output-dir ../../sample-pictures-out --clean-steps
```

## To develop the website locally

To run just on your computer:

```bash
pnpm dev
```

To run on your network:

```bash
pnpm dev:host
```

Note: these will enable hot reloading for the website code but not for the extraction pipeline. If you make changes to the extraction pipeline, re-run the script.

## To build the website and host a production preview

To run just on your computer:

```bash
pnpm preview
```

To run on your network:

```bash
pnpm preview:host
```

## To publish to GitHub Pages

Create a PR merging `main` into `production` branch. Once this is merged, `deploy.yml` will automatically run and do the following:

1. Publish the website to GitHub Pages
2. Create a release in GitHub
3. Bump the minor versions in the `package.json` files

# Python development instructions

## Python Setup

```bash
cd packages/gbcam-extract-py
python -m venv .venv

# Unix-based
source .venv/bin/activate
# Windows
./.venv/Scripts/activate

pip install -r requirements.txt
```

### To regenerate `requirements.txt`

Inside `packages/gbcam-extract-py` after [activating the `.venv`](#python-setup):

```bash
pip install pipreqs
pipreqs . --ignore=.venv --encoding=utf8 --force
```

## To run the extraction pipeline locally in Python

Inside `packages/gbcam-extract-py` after [activating the `.venv`](#python-setup):

To generate the sample output pictures from the sample input pictures, run the script as follows:

```bash
python gbcam_extract.py --dir ../../sample-pictures --output-dir ../../sample-pictures-out-py --clean-steps
```

To generate the sample images for each step for just one sample input picture, run the script as follows:

```bash
python gbcam_extract.py ../../sample-pictures/20260313_213430.jpg --output-dir ../../sample-pictures-out-py
```

The following command-line arguments are reasonably useful:

- `--help`, `-h` - print information about the script, all its command-line arguments, its process for transforming images, and usage examples
- `<file-paths>` - provide file paths or globs to input files to transform
- `--dir` - specify directory to search for input files
- `--output-dir` - specify directory to put output files
- `--start <step>` - specify a starting step at which to process files (useful for adjusting individual steps in the process). See the printed help info for options. Defaults to `warp` (the first step)
- `--end <step>` - specify a final step for processing files. See the printed help info for options. Defaults to `quantize` (the last step)
- `--debug` - outputs information about the process of transforming the image and outputs many images detailing the transformation process
- `--clean-steps` - after processing images, delete the intermediate step out files (or put them in the `debug` folder if `--debug` was provided)

See the help information for additional details like usage examples (please note that the help info is AI-generated):

```bash
python gbcam_extract.py --help
```

## To Test

Inside `packages/gbcam-extract-py` after [activating the `.venv`](#python-setup):

To run the full test suite that regenerates all the images checked into this repo, run the following:

```bash
python run_tests.py
```

This will run commands like the following:

```bash
python gbcam_extract.py --dir ../../sample-pictures --output-dir ../../sample-pictures-out-py --clean-steps --debug
python test_pipeline.py --input "../../test-input/zelda-poster-1.jpg" --reference "../../test-input/zelda-poster-output-corrected.png" --output-dir ../../test-output-py/zelda-poster-1 --keep-intermediates
python test_pipeline.py --input "../../test-input/zelda-poster-2.jpg" --reference "../../test-input/zelda-poster-output-corrected.png" --output-dir ../../test-output-py/zelda-poster-2 --keep-intermediates
python test_pipeline.py --input "../../test-input/thing-1.jpg" --reference "../../test-input/thing-output-corrected.png" --output-dir ../../test-output-py/thing-1 --keep-intermediates
python test_pipeline.py --input "../../test-input/thing-2.jpg" --reference "../../test-input/thing-output-corrected.png" --output-dir ../../test-output-py/thing-2 --keep-intermediates
```

To run a unit test to test the accuracy of the output, gather the following:

- Input image: an input picture of a Game Boy Camera picture [as described above](#taking-pictures-that-work-with-this-script) (e.g. `test-input/zelda-poster-1.jpg`)
- Reference image: a perfectly digitized 128x112 reproduction of the input Game Boy Camera picture (e.g. `test-input/zelda-poster-output-corrected.png`)

Then run the following:

```bash
python test_pipeline.py --input "../../test-input/zelda-poster-1.jpg" --reference "../../test-input/zelda-poster-output-corrected.png" --output-dir ../../test-output-py/zelda-poster-1 --keep-intermediates
```

# Known issues

- Conversion does not preserve the image with 100% accuracy
- Debug mode in the website is completely untested

# Roadmap

- Accuracy improvements
- Initial crop from phone picture to cropped and rotated image that is input to "warp" step
