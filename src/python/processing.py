import numpy as np


def data_processing(pose_landmarks,mtx,path_length):
    print("Processing joint data....")
    time = pose_landmarks["length"]
    speed = path_length / time
    if path_length > 0:
        dist = 0
    else:
        dist = -path_length
    
    for t, frame in enumerate(pose_landmarks["frames"]):
        new_prediction = []
        new_joint = {}
        for i, landmark in enumerate(frame["prediction"].pose_landmarks[0]):
            # calculate image center
            center = (pose_landmarks["video_width"]/2, pose_landmarks["video_height"]/2)
            # convert normalized coordinates to pixel coordinates
            pixel_x = center[0] - landmark.x*pose_landmarks["video_width"]
            pixel_y = center[1] - landmark.y*pose_landmarks["video_height"]
            # convert pixel coordinates to real world coordinates
            est_z = dist + speed*(frame["timestamp"])
            # calculate joint depth for this joint
            joint_z = est_z + frame["prediction"].pose_world_landmarks[0][i].z + 1
            new_joint["z"] = joint_z
            new_joint["x"] = (pixel_x)*new_joint["z"]/mtx[0,0]
            new_joint["y"] = (pixel_y)*new_joint["z"]/mtx[1,1]
            
            # add new joint data in prediction set
            new_prediction.append(new_joint)
        # replace prediction set with new set
        pose_landmarks["frames"][t]["prediction"] = new_prediction
    return pose_landmarks

    

            

