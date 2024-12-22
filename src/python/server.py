from flask import Flask, request, jsonify
import numpy as np
from calibration import calibrate
from pose import blazepose
from processing import data_processing
from checkerboard import video_split
from undistort import videoprocessing
from gait_measures import gait_measures
from random import randint
import os
from flask_cors import CORS
app = Flask(__name__)
CORS(app)
cdata = {
    'mtx': None,
    'dist': None,
    'rvecs': None,
    'tvecs': None
}
@app.route('/')
def home():
    return 'Welcome to the 3D HPE Gait Analysis API'

@app.route('/health')
def health_check():
    return jsonify({'status': 'healthy'})

@app.route('/cv', methods=['POST'])
def cv_upload_data():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    os.makedirs(os.getcwd() + '/../../uploads/', exist_ok=True)
    filename = "cv_"+str(randint(1000,9999))+".mp4"
    video_path = os.getcwd() + '/../../uploads/' + filename
    with open(video_path, 'wb') as f:
        f.write(file.read())
        f.close()
    # split the checkerboard video into frames
    checkerboard_path = video_split(video_path)
    # calibrate the camera
    cdata['mtx'], cdata['dist'], cdata['rvecs'], cdata['tvecs'] = calibrate(8, 6, checkerboard_path)
    return jsonify({'message': 'Calibration successful'})
    # Enable CORS for all routes


@app.route('/gv', methods=['POST'])
def gv_upload_data():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    filename = "gv_"+str(randint(1000,9999))+".mp4"
    os.makedirs(os.getcwd() + '/../../uploads/', exist_ok=True)
    video_path = os.getcwd() + '/../../uploads/' + filename
    with open(video_path, 'wb') as f:
        f.write(file.read())
        f.close()
    # undistort the gait video and obtain new camera matrix
    output_path = os.getcwd() + '/../../undistorted_frames/' + filename
    # REMOVE ASAP
    cdata['mtx'], cdata['dist'], cdata['rvecs'], cdata['tvecs'] = calibrate(8, 6, "C:\\Users\\pc\\projects\\3DHPE_GaitAnalysis\\frames")
    #ok continue
    cdata['mtx'] = videoprocessing(cdata['mtx'], cdata['dist'], video_path, "../../undistorted_frames/", filename)
    # obtain joint data from the undistorted gait video and get path length
    joint_data = blazepose(output_path)
    path_length = float(request.headers['Path-Length'])
    # process the joint data into an appropriate format
    new_joint_data = data_processing(joint_data, cdata['mtx'], path_length)
    # extract gait measures from the processed joint data
    measures = gait_measures(new_joint_data, 'measures.xlsx')
    return jsonify({'message': 'Processing successful', 'measures': measures})
# Add more routes here as needed

if __name__ == '__main__':
    print("Starting server...")
    app.run(debug=True, host='localhost', port=3000)