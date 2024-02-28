# File Description : AI Stuff. 
# Input: 2 videos of people dancing (1 student, 1 professional)
# Output: 
    # 1. Score 
    # 2. Feedback 


import os
from pathlib import Path

# Functions 



# This is the main function


def process_videos(videoPath1 : str , videoPath2 : str):

    print("Processing Videos")



    # We want the video to be .mp4

    videoPath1 = convertToMp4(videoPath1)
    videoPath2 = convertToMp4(videoPath2)

    
    
    # We want to get the "Frame Angles"
        # Frame Angles -- Array of different angles in the body (btwn shoulder elbow wrist)

    frames1 , imageName1 = get_frames_angles(videoPath1)
    frames2 , imageName2 = get_frames_angles(videoPath2)



# Datatype -- int, float, string, bool

from moviepy.editor import VideoFileClip

def convertToMp4(path) -> str:

    # funnyVideo.mov

    _, extension = os.path.splitext(path)

    if extension != '.mp4':
        mp4_path = Path(path).stem + '.mp4' 
        clip = VideoFileClip(path)
        clip.write_videofile(mp4_path, codec='libx264')

        return mp4_path
    

    return path



# Function -- get_frame_angles

# Input: video_path : string
# Output: For each frame in the video, we're going to output the angles 
# [

    #Frame1,
    # Frame2,
    # Frame1000,

# ]


# student : tuple = ("ShengLin" , 4.0) 



# video_path = "coolVideo.mp4"


import cv2

def get_frames_angles(video_path) -> tuple:


    frames : list = []
    image_name : str = ""


    pose , poseDrawing = initializePoseTools()

    
    # frames = [
        
    #     [45.6, 78.1, 98.1],     # 1
    #      [5.6, 34.1, 98.1],     # 2
    #       [9.6, 78.1, 98.1],   
    #        [45.6, 78.1, 98.1],                      
    # ]


    cap = cv2.VideoCapture(video_path)


    with pose.Pose(
        min_detection_confidence = 0.5,
        min_tracking_confidence = 0.5
    ) as poseModel:
        
        while cap.isOpened():

            success , frame = cap.read()

            # Create an if statement that says: If it's not successful, then print out "Could not read the frame" 

            if not success:

                print("Could not read the frame")
                break

            

            # If the program gets to here


            # 1) Pose Process the Frame 
                # Annotate the image with the lines and circles on body parts
                # Give us the Landmark Results -- we need this to get the angles
            

            # Put the pose model into our image
            pose_processed_frame , landmark_results = pose_process_image(frame, poseModel)


            # 2) Get the Angles



            # 3) Save the frame and save the angles





    return (frames, image_name)


import mediapipe as mp
from mediapipe.python.solutions.pose import Pose

def initializePoseTools():

    pose = mp.solutions.pose
    
    mp_drawing = mp.solutions.drawing_utils


    return pose, mp_drawing

def pose_process_image(openCVFrame, poseModel : Pose ) -> tuple:
    
    
    # OpenCV processes images in BGR

    # Mediapipe processes images in RGB
    

    # RGB <------  BGR
    rgbImage = cv2.cvtColor(openCVFrame, cv2.COLOR_BGR2RGB)
    
    # Since it's in mediapipe image format now, we can use the pose model


    landmark_results = poseModel.process(rgbImage)
    
    if landmark_results.pose_landmarks:

        for id, landmark in enumerate(landmark_results.pose_landmarks.landmark):

            print(id)
            print(f"x: {landmark.x}")
            print(f"y: {landmark.y}")
            print(f"z: {landmark.z}")

            print()
    
    
    pose_processed_frame = None

    


    return (pose_processed_frame, landmark_results)

    # .mov, mp4 -- video extensions
    # .pdf .txt -- text extensions



process_videos("golfVideo.mp4" , "golfVideo.mp4")























