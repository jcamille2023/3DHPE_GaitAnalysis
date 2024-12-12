import matlab.engine
import asyncio
import os
from openpyxl import Workbook



def save_joint_data(data,filename):
    print("Saving joint data....")
    wb = Workbook()
    s = wb.active
    s.append(["timestamp","joint","x","y","z"])
    for frame in data["frames"]:
        for tdx, landmark in enumerate(frame["prediction"]):
            s.append([frame["timestamp"], # timestamp
                    tdx, # joint number
                    landmark["x"], # scaled x coord
                    landmark["y"], # scaled y coord
                    landmark["z"]]) # absolute z coord
    os.makedirs(os.path.join(os.getcwd(),"\\predictions\\"), exist_ok=True)
    wb.save(os.path.join(os.getcwd(),"\\predictions\\") + filename)
    
    return os.path.join(os.getcwd(),"..\\..\\predictions\\") + filename

def gait_measures(pose_landmarks,filename):
    data_path = save_joint_data(pose_landmarks,filename)
    print("Extracting gait measures with MATLAB....")
    eng = matlab.engine.start_matlab()
    eng.addpath(os.path.join(os.getcwd(),"\\src\\matlab\\gait_measures.m"))
    result = dict(eng.feval("gait_measures",data_path))
    return result





