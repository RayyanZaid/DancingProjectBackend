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


            image = draw_landmarks(landmark_results, poseDrawing, pose, pose_processed_frame)

            # Test
            # cv2.imshow("Frame" , image)
            # cv2.waitKey(1)

            # 2) Get the Angles

            h,w, _ = image.shape 
            
            angles = get_angles_for_each_frame(pose, landmark_results, image, h, w)

            # 3) Save the frame and save the angles





    return (frames, image_name)


import mediapipe as mp
from mediapipe.python.solutions.pose import Pose



def get_angles_for_each_frame(pose, landmark_results, image, h, w):
    
    # 6 angles
    angles = []


    angle = 56

    angles.append(angle)

    angle = 170

    angles.append(angle)




def draw_landmarks(results, mp_drawing, mp_pose, image):
    # for idx (index), x (value) in enumerate(_____):   \\storing both the index and the value
    # work w/both variables simultaneously; requires
    for idx, landmark in enumerate(results.pose_landmarks.landmark):
        # we care about 11-16 and 23-28
        if idx in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 17, 18, 19, 20, 21, 22, 29, 30, 31, 32]:
            results.pose_landmarks.landmark[idx].visibility = 0  # remove visibility of specific landmarks

    # draw landmarks
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
                              # customize color, etc
                              mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

    return image


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



    # After this line, rgbImage now has the lines and marks
    landmark_results = poseModel.process(rgbImage)
    
    if landmark_results.pose_landmarks:

        for id, landmark in enumerate(landmark_results.pose_landmarks.landmark):

            print(id)
            print(f"x: {landmark.x}")
            print(f"y: {landmark.y}")
            print(f"z: {landmark.z}")

            print()
    
    
    # BGR  <-------- RGB
    openCVFrame = cv2.cvtColor(rgbImage , cv2.COLOR_RGB2BGR)

    
    


    return (openCVFrame, landmark_results)

    # .mov, mp4 -- video extensions
    # .pdf .txt -- text extensions



process_videos("golfVideo.mp4" , "golfVideo.mp4")























