# CycleGan_on_CartoonSeries
This project is based on https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix


# VideoSync: Advanced Video Frame Processing and Perceptual Analysis Pipeline

This repository contains a sophisticated pipeline for video frame extraction, perceptual loss computation using a modified VGG16 model, and video reconstruction with original and generated frame pairs. Designed for tasks like style transfer or domain adaptation (e.g., CycleGAN/pix2pix), the project processes videos into frame pairs, prepares a feature extractor for loss computation, and reconstructs videos for side-by-side comparison of original and generated content.

## Project Overview

- **Objective**: Extract frames from MP4 videos, process them into left/right pairs for GAN input, compute perceptual losses using a trimmed VGG16 model, and reconstruct AVI videos combining original and generated ("fake") frames.
- **Technologies**: Utilizes Python with PyAV, NumPy, PyTorch (torchvision), OpenCV, Matplotlib, Pillow, SciPy, and Pickle for efficient video processing, neural network operations, and data serialization.
- **Context**: Likely part of a larger GAN-based workflow (e.g., CycleGAN for domain translation like "Winnie" style transfer), handling video frames as paired crops for training or inference.

## Dependencies

Install the required libraries using pip:

```bash
pip install av numpy torch torchvision pillow matplotlib scipy opencv-python ipython
```

- **Python Version**: Tested with Python 3.7+ (based on notebook metadata).
- **Notes**: Pretrained VGG models require internet access. Ensure directories like `./mp4movie/`, `./Data/`, and `./pytorch-CycleGAN-and-pix2pix/datasets/` exist or adjust paths accordingly.

## Notebooks Explanation

The pipeline consists of three Jupyter notebooks, executed sequentially for the complete workflow: frame extraction → model preparation → video reconstruction.

### 1. `frames.ipynb` - Video Frame Extraction

Extracts frames from an MP4 video, crops them into left/right halves, and saves them as images for GAN training or processing.

#### Key Components:
- **Imports**:
  - `av`: Video decoding.
  - `numpy`: Array manipulation.
  - `pickle`: Data serialization.
  - `matplotlib.pyplot`: Visualization.
  - `IPython.display`: Progress output management.
  - `scipy.signal`: Defines `color_to_bw` and `sobel_kernel` (unused—possibly for future grayscale/edge detection).
- **Functions**:
  - `SaveArray(name, array)`: Saves NumPy arrays to pickle in `./Data/`.
  - `OpenArray(name, dump_num)`: Loads pickle files from `./Data/`.
  - `GetFrames(name, frames_num)`:
    - Opens video from `./mp4movie/{name}.mp4`.
    - Skips frames based on desired output count.
    - Reformats frames to RGB, crops 256x256 left/right sections from center.
    - Saves JPGs to `./pytorch-CycleGAN-and-pix2pix/datasets/winnie2winnie/testB/` for GAN input.
    - Processes frames 50000–52500 (hardcoded—modify as needed).
    - Shows progress with `clear_output`.
- **Additional**:
  - Runs `!ls` to list files.
  - Visualizes two frames (indices 240, 241) side-by-side.

#### Usage:
- Execute `GetFrames('video_name', num_frames)` to extract frames.
- Example: Processes ~500 frames, saving them as JPGs.
- Output: JPGs in dataset directory; pickled arrays if saved.

#### Purpose:
Prepares video frames as image pairs for GAN training (e.g., style transfer).

### 2. `VGG_prepare.ipynb` - Perceptual Loss Model Preparation

Configures a modified VGG16 model for computing perceptual losses, essential for improving GAN output quality beyond pixel-level differences.

#### Key Components:
- **Imports**:
  - `torchvision`, `models`: VGG16 access.
  - `PIL`: Image handling.
  - `numpy`, `torch`: Tensor operations.
  - `pickle`: Model serialization.
- **Model Setup**:
  - Loads pretrained `vgg16_bn`.
  - Modifies:
    - Retains first 10 `features` layers (three convolutions).
    - Replaces `classifier` and `avgpool` with empty sequences.
  - Freezes layers (`layer.trainable = False`).
  - Saves as `VGG_3conv.pickle`; loads for verification.
- **Image Processing**:
  - Loads `1790_fake.png` and `1790_real.png`.
  - Converts to tensors ([C, H, W]).
  - Extracts features via forward pass.
- **Loss Computation**:
  - Computes L1 loss between VGG features of fake/real images (perceptual loss).
  - Compares with pixel-level L1 loss.

#### Usage:
- Run to save modified VGG model.
- Example: `loss(res_fake, res_real)` computes perceptual loss.
- Output: Pickled model; loss values.

#### Purpose:
Provides a feature extractor for perceptual loss in GAN training.

### 3. `video_maker.ipynb` - Video Reconstruction

Reconstructs AVI videos from original and generated frame pairs, with visualization and image-saving utilities.

#### Key Components:
- **Imports**:
  - `os`, `cv2`: Video writing.
  - `pickle`: Serialization.
  - `matplotlib.pyplot`, `numpy`: Visualization/arrays.
  - `torchvision`: Unused (possibly for transforms).
  - `IPython.display`, `PIL`: Progress and image saving.
- **Functions**:
  - `make_video(pairs_list1, pairs_list2, name)`:
    - Creates AVI video (15 FPS, 512x512).
    - Stacks left/right frames horizontally, originals/fakes vertically.
  - `save_from_pickle(name, path='')`:
    - Saves frame pairs as PNGs (e.g., `0_left.png`).
  - `SaveArray(name, array)`: Saves lists to pickle.
  - `make_pickle(num1, num2, path='./images/', name='_fake_B.png')`:
    - Creates pickle from image pairs (e.g., GAN outputs).
- **Additional**:
  - Loads `USA2RU_orig.pkl` and `USA2RU_forward.pkl`.
  - Visualizes sample pairs.
  - Example: Creates `usa_winni.avi` from frame slices.
  - Saves images to `./testA/` or builds pickles (e.g., frames 1000–2500).

#### Usage:
- Run `make_video(orig, fake, 'output.avi')` for video output.
- Use `save_from_pickle` or `make_pickle` for image/pickle handling.
- Output: AVI videos; PNGs; pickled frame lists.

#### Purpose:
Reassembles GAN outputs with originals for visual comparison.

## Workflow

1. **Frame Extraction** (`frames.ipynb`): Extract and crop frames for GAN input.
2. **Model Preparation** (`VGG_prepare.ipynb`): Create VGG feature extractor.
3. **GAN Processing** (External): Generate fake frames (assumed via pix2pix/CycleGAN).
4. **Video Reconstruction** (`video_maker.ipynb`): Combine original/fake frames into videos.

## Notes
- **Hardcoded Values**: Adjust video paths, frame ranges (e.g., 50000–52500), and directories.
- **Performance**: Timed sections (e.g., VGG forward pass) included for optimization.
- **Extensions**: Integrate with CycleGAN/pix2pix for full pipeline.
- **Issues**: Open a GitHub issue for support.
