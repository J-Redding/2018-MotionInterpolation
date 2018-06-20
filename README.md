Motion interpolation project.
Uses a trained Unet convolutional neural network to predict middle frames in a video.
Based on neil454's Deep Motion implementation, found here:
https://github.com/neil454/deep-motion

Requires the CNN weights to be downloaded from here:
https://github.com/neil454/deep-motion/releases/download/0.1/weights_unet2_finetune_youtube_100epochs.hdf5

Takes in a 24 frames per second video.
24_to_48 doubles the frame rate to 48 fps.
12_to_24 slows the video down to 12 fps and then restores the video to 24 fps using motion interpolation.
