# File Description : AI Stuff. 
# Input: 2 videos of people dancing (1 student, 1 professional)
# Output: 
    # 1. Score 
    # 2. Feedback 


import os
from pathlib import Path


# Machline Learning Model Imports

from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.model_selection import ParameterGrid


# Functions 



# This is the main function


def process_videos(videoPath1 : str , videoPath2 : str):

    print("Processing Videos")




    # 1) TO MP4

    # We want the video to be .mp4

    videoPath1 = convertToMp4(videoPath1)
    videoPath2 = convertToMp4(videoPath2)

    
    
    # We want to get the "Frame Angles"
        # Frame Angles -- Array of different angles in the body (btwn shoulder elbow wrist)


    # 2) Getting our data

    video1FrameData , imageName1 = get_frames_angles(videoPath1)
    video2FrameData , imageName2 = get_frames_angles(videoPath2)


    # 3) Using Machine Learning

    
    video1KeyFrames = []
    video2KeyFrames = []

    public_urls = get_image_urls(video1FrameData, video2FrameData, imageName1, imageName2, video1KeyFrames, video2KeyFrames)




# 3) Using Machine Learning

def get_image_urls(video1FrameData, video2FrameData, imageName1, imageName2, video1KeyFrames, video2KeyFrames):
    

    # 1) Use the Student Data (video1FrameData) and create the clusters

    student_cluster = get_cluster(video1FrameData)



def get_cluster(video1FrameData):
    
    numClusters = kmean_hyper_param_tuning(video1FrameData)

    X = np.array(video1FrameData)

    # Create KMeans model (best line of fit) with 'n' clusters using our video1 data
    kmeans_1 = KMeans(n_clusters=numClusters).fit(X)
    



# The NUMBER of key frames
def kmean_hyper_param_tuning(video1FrameData):
    
    parameters = []


    for i in range(2,31):
        parameters.append(i)

    parameter_grid = ParameterGrid({'n_clusters' : parameters})
    
    # go through params in parameter_grid (check which one is the best)
    
    best_score = -1
    best_grid  = {}

    kmeans_model = KMeans()

    for p in parameter_grid:

        kmeans_model.set_params(**p)
        kmeans_model.fit(video1FrameData)

        ss = metrics.silhouette_score(video1FrameData, kmeans_model.labels_)

        print("Parameter:", p, 'Score:', ss)

        if ss > best_score:
            best_score = ss
            best_grid = p

    return best_grid['n_clusters']


# 2) Getting Data 

def make_directory(name : str):

    # if the directory does NOT exist, create it
    if not os.path.isdir(name):
        os.mkdir(name)


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


    # main part of function is frames
    frame_angles : list = []
    
    basename = os.path.basename(video_path)
    image_name , _ = os.path.splitext(basename)

    make_directory(image_name)


    pose , poseDrawing = initializePoseTools()

    
    # frames = [
        
    #     [45.6, 78.1, 98.1],     # 1
    #      [5.6, 34.1, 98.1],     # 2
    #       [9.6, 78.1, 98.1],   
    #        [45.6, 78.1, 98.1],                      
    # ]


    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print("FPS: " , fps)


    img_count = 1

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
            
            angles : list = get_angles_for_each_frame(pose, landmark_results, image, h, w)

            # 3) Save the frame and save the angles

            # frames = [
        
            #     [45.6, 78.1, 98.1],     # 1
            #      [5.6, 34.1, 98.1],     # 2
            #       [9.6, 78.1, 98.1],   
            #        [45.6, 78.1, 98.1],                      
            # ]

            frame_angles.append(angles)

                            #   golfVideo/1
            imageFilePath = f"{image_name}/{img_count}.jpg"
            img_count += 1

            # At the path, save the image
            cv2.imshow('Video' , image)
            cv2.waitKey(1)
            cv2.imwrite(imageFilePath, image)



    return (frame_angles, image_name)


import mediapipe as mp
from mediapipe.python.solutions.pose import Pose

import numpy as np

def calculate_angle(a,b,c):
    
    # a = b = c = [x,y]

    a = np.array(a) # p1
    b = np.array(b) # p2
    c = np.array(c) # p3


    # USE arctan to calculate the angle

    radians = np.arctan2(c[1] - b[1], c[0] - b[0])   -    np.arctan2(a[1] - b[1], a[0] - b[0])

    degrees = np.abs(radians * 180.0 / np.pi)

    if degrees > 180.0:

        degrees = 360 - degrees

    return round(degrees,1)



def draw_angle(actualCoordinate : tuple, image, angle):

    

    angleStr = str(angle)

    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.4
    color = (255,255,255)
    thickness = 1

    
    drawnImage = cv2.putText(image,angleStr, actualCoordinate,font,fontScale,color,thickness)

    return drawnImage




# function name : plot_angle
# parameters : p1, p2, p3, landmark_results, image, h , w
def plot_angle(p1,p2,p3,landmark_result,image,h ,w):

    landmark_result = landmark_result.pose_landmarks.landmark
    a = [   landmark_result[p1].x   ,   landmark_result[p1].y        ]

    b = [     landmark_result[p2].x ,  landmark_result[p2].y   ]
    c = [     landmark_result[p3].x ,   landmark_result[p2].y  ]

    # Step 1 : Calculate the angle in degrees

    angle = calculate_angle(a,b,c)

    # angle = 134.6

    # Step 2 : Draw the angle on the image

    # the middle angle (b), angle calculation, image that we're gonna draw on


    actualXCoordinate = int(b[0] * w)
    actualYCoordinate = int(b[1] * h)

    actualCoordinate = tuple(np.multiply(b,[w,h]).astype(int))


    actualCoordinate = (actualXCoordinate,actualYCoordinate)


    drawnImage = draw_angle(actualCoordinate, image, angle)
    
    return angle, drawnImage



def get_angles_for_each_frame(mp_pose, landmarks, image, h, w):
    
    # 6 angles
    angles = []
    val = 50

    # 3 points
        # 1. Left Shoulder
        # 2. Left Elbow
        # 3. Left Wrist
    
    angle, image = plot_angle(mp_pose.PoseLandmark.LEFT_SHOULDER.value,
                              mp_pose.PoseLandmark.LEFT_ELBOW.value,
                              mp_pose.PoseLandmark.LEFT_WRIST.value, landmarks, image, h, w + val)
    angles.append(angle)
    angle, image = plot_angle(mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
                              mp_pose.PoseLandmark.RIGHT_ELBOW.value,
                              mp_pose.PoseLandmark.RIGHT_WRIST.value, landmarks, image, h, w - val)
    angles.append(angle)
    angle, image = plot_angle(mp_pose.PoseLandmark.LEFT_HIP.value,
                              mp_pose.PoseLandmark.LEFT_KNEE.value,
                              mp_pose.PoseLandmark.LEFT_ANKLE.value, landmarks, image, h, w + val)
    angles.append(angle)
    angle, image = plot_angle(mp_pose.PoseLandmark.RIGHT_HIP.value,
                              mp_pose.PoseLandmark.RIGHT_KNEE.value,
                              mp_pose.PoseLandmark.RIGHT_ANKLE.value, landmarks, image, h, w - val)
    angles.append(angle)



    angle, image = plot_angle(mp_pose.PoseLandmark.LEFT_SHOULDER.value,
                              mp_pose.PoseLandmark.LEFT_HIP.value,
                              mp_pose.PoseLandmark.LEFT_KNEE.value, landmarks, image, h, w + val)
    angles.append(angle)



    angle, image = plot_angle(mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
                              mp_pose.PoseLandmark.RIGHT_HIP.value,
                              mp_pose.PoseLandmark.RIGHT_KNEE.value, landmarks, image, h, w - val)
    angles.append(angle)



    angle, image = plot_angle(mp_pose.PoseLandmark.LEFT_WRIST.value,
                             mp_pose.PoseLandmark.LEFT_SHOULDER.value,
                             mp_pose.PoseLandmark.LEFT_HIP.value, landmarks, image, h, w + val)
    angles.append(angle)




    angle_wrist_shoulder_hip_right, image = plot_angle(mp_pose.PoseLandmark.RIGHT_WRIST.value,
                                                       mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
                                                       mp_pose.PoseLandmark.RIGHT_HIP.value, landmarks, image, h,
                                                       w - val)
    
    angles.append(angle_wrist_shoulder_hip_right)

    


    # cv2.imshow('Hopefully this works' , image)
    # cv2.waitKey(1)

    return angles

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
    
    # if landmark_results.pose_landmarks:

    #     for id, landmark in enumerate(landmark_results.pose_landmarks.landmark):

    #         print(id)
    #         print(f"x: {landmark.x}")
    #         print(f"y: {landmark.y}")
    #         print(f"z: {landmark.z}")

    #         print()
    
    
    # BGR  <-------- RGB
    openCVFrame = cv2.cvtColor(rgbImage , cv2.COLOR_RGB2BGR)

    
    


    return (openCVFrame, landmark_results)

    # .mov, mp4 -- video extensions
    # .pdf .txt -- text extensions



process_videos("golfVideo.mp4" , "golfVideo.mp4")























