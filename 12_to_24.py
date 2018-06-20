import os
import time
import numpy as np
from scipy.misc import imresize
import cv2
from FI_unet import get_unet

os.environ['KERAS_BACKEND'] = "tensorflow"
os.environ['THEANO_FLAGS'] = "device=gpu0, lib.cnmem=0.85, optimizer=fast_run"

#Returns a numpy array of the video frames for a provided video
def get_video_array(video_path):
    cap = cv2.VideoCapture(video_path)
    number_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    video_array = np.zeros(shape = (number_frames, 128, 384, 3), dtype = "uint8")
    #Get the video frames as a numpy array
    for frames in range(number_frames):
        ret, frame = cap.read()
        video_array[frames] = imresize(frame, (128, 384))

    return video_array

#Returns the frames per second for a provided video
def get_fps(video_path):
    cap = cv2.VideoCapture(video_path)
    return cap.get(cv2.CAP_PROP_FPS)

#Reduces the specified video to six frames per second
#Used to give us an initial video with low framerate
#This low framerate video is scaled down to have dimensions of 128 x 384
#The videos need to initially be 128 x 384 by demand of the neural network; it only returns these videos
def set_video_twelve_frames_per_second(video_array):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    #Set up the output video
    out = cv2.VideoWriter("smallVideos/12fps.avi", fourcc, 12, (384, 128))
    #Write the frames to the output video
    for frames in range(len(video_array)):
        #We know that the original video was 24 fps
        #We know that the original video went for 15 seconds
        #That means there were 360 frames in the original video
        #We want 90 frames in the new video
        #So we take write out every eighth frame
        #(360 / 90 = 4)
        if frames % 2 == 0:
            out.write(video_array[frames])

#Upscales an existing 128 x 384 video to 720 x 1280
def upscale_video(video_array, video_save_path, fps):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    #Set up the output video
    out = cv2.VideoWriter("Videos/" + video_save_path, fourcc, fps, (1280, 720))
    #Write the frames to the output video
    for frames in range(len(video_array)):
        frame = video_array[frames]
        #Resize the current frame to 720 x 1280
        resize = cv2.resize(frame, (1280, 720), interpolation = cv2.INTER_LINEAR)
        out.write(resize)

#Doubles the framerate for a given video
#Takes in a video as a numpy array of frames
#Returns a numpy array of the video with doubled framerate
#The in-between frames are fabricated through the neural network
def double_fps(video_array):
    #Get the NN model
    model = get_unet((6, 128, 384))
    #Assign the weights
    #The weights are pre-determined, according to pre-training
    model.load_weights("weights_unet2_finetune_youtube_100epochs.hdf5")
    doubled_video_array = []
    #Attach the first frame into the new array as an initial frame
    doubled_video_array.append(video_array[0])
    #For each existing frame
    for frames in range(1, len(video_array)):
        #Get the predicted frame from the model
        prediction = model.predict(np.expand_dims(np.transpose(np.concatenate((video_array[frames - 1], video_array[frames]), axis = 2) / 255., (2, 0, 1)), axis = 0))
        #Attach the predicted frame
        #(Frame n.5)
        doubled_video_array.append((np.transpose(prediction[0], (1, 2, 0)) * 255).astype("uint8"))
        #Attach the regular frame
        #(n + 1)
        doubled_video_array.append(video_array[frames])

    return np.asarray(doubled_video_array)

#Saves a video array as a video
#These videos have dimensions of 128 x 384
#They are later scaled up to 720p
#The videos need to initially be 128 x 384 by demand of the neural network; it only returns these videos
def save_small_video(video_array, video_save_path, fps):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    #Set up the output video
    out = cv2.VideoWriter("smallVideos/" + video_save_path, fourcc, fps, (384, 128))
    #Write the frames to the output video
    for frames in range(len(video_array)):
        out.write(video_array[frames])

#The main function takes in a sample video 
#The sample video is reduced to six frames per second
#The reduced video's frames per second are then doubled to 12, 24, and 48 frames per second
def main():
    #This is the path to the original video
    #The original video was 15 seconds long and had 24 fps
    video_path = "video.mp4"
    video_array = get_video_array(video_path)
    fps = get_fps(video_path)
    set_video_twelve_frames_per_second(video_array)
    video_path = "smallVideos/12fps.avi"
    video_array = get_video_array(video_path)
    upscale_video(video_array, "12fps.avi", 12)
    start_time = time.time()
    video_array = double_fps(video_array)
    end_time = time.time()
    print(end_time - start_time)
    save_small_video(video_array, "24fps.avi", 24)
    upscale_video(video_array, "24fps.avi", 24)
    start_time = time.time()
    video_array = double_fps(video_array)
    end_time = time.time()
    print(end_time - start_time)
    save_small_video(video_array, "48fps.avi", 48)
    upscale_video(video_array, "48fps.avi", 48)

main()