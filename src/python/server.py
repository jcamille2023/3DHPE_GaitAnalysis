import asyncio
import websockets
import json
import base64
import os
from checkerboard import video_split
from datetime import datetime
from calibration import calibrate
from undistort import videoprocessing
from processing import data_processing
from gait_measures import gait_measures
from pose import blazepose
class VideoWebSocketServer:
    def __init__(self, host='localhost', port=3001):
        self.host = host
        self.port = port
        self.connected_clients = set()
        self.video_upload_dir = 'uploaded_videos'
        self.camera_params = {}
        
        
        # Create directory for video uploads if it doesn't exist
        if not os.path.exists(self.video_upload_dir):
            os.makedirs(self.video_upload_dir)

    async def send_status(self, websocket, status_message):
        """Send a status event to a specific client."""
        status_event = {
            'type': 'status',
            'message': status_message,
            'timestamp': datetime.now().isoformat()
        }
        await websocket.send(json.dumps(status_event))

    async def send_gait_data(self, websocket, gait_metrics):
        """Send gait data event to a specific client."""
        gait_event = {
            'type': 'gait_data',
            'metrics': gait_metrics,
            'timestamp': datetime.now().isoformat()
        }
        await websocket.send(json.dumps(gait_event))

    async def handle_checkerboard_video_upload(self, websocket, data):
        """
        Handle video upload from client.
        Expects base64 encoded video data.
        """
        try:
            # Decode base64 video data
            video_data = base64.b64decode(data)
            
            # Generate unique filename
            filename = f'{self.video_upload_dir}/checkerboard_video_{datetime.now().strftime("%Y%m%d_%H%M%S")}.mp4'
            
            # Save video file
            with open(filename, 'wb') as video_file:
                video_file.write(video_data)
            
            # Send success status
            await self.send_status(websocket, f'Video saved: {filename}')
            await self.send_status(websocket, f'Processing checkerboard video..')
            await self.send_status(websocket, f'Conducting camera calibration...')
            
            path = video_split(filename)
            
            mtx, dist, rvecs, tvecs = calibrate(8,6,path)
            self.camera_params["mtx"] = mtx
            self.camera_params["dist"] = dist
            self.camera_params["rvecs"] = rvecs
            self.camera_params["tvecs"] = tvecs
            await self.send_status(websocket, f'Camera calibration complete')
            await self.send_status(websocket, f'success')

            return filename
        except Exception as e:
            # Send error status
            await self.send_status(websocket, f'Video upload failed: {str(e)}')
            return None
    async def handle_gait_video_upload(self, websocket, data, path_length):
        """
        Handle video upload from client.
        Expects base64 encoded video data.
        """
        try:
            # Decode base64 video data
            video_data = base64.b64decode(data)
            
            # Generate unique filename
            filename = f'{self.video_upload_dir}/gait_video_{datetime.now().strftime("%Y%m%d_%H%M%S")}.mp4'
            
            # Save video file
            with open(filename, 'wb') as video_file:
                video_file.write(video_data)
            
            # Send success status
            await self.send_status(websocket, f'Gait video saved: {filename}')
            await self.send_status(websocket, f'Applying camera matrix and distortion coefficients...')
            self.camera_params["mtx"] = videoprocessing(self.camera_params["mtx"],self.camera_params["dist"],filename,self.video_upload_dir,filename[:-3]+"undistorted.mp4")
            await self.send_status(websocket, f'Undistortion complete')
            await self.send_status(websocket, f'Running pose estimation...')
            joint_data = blazepose(self.video_upload_dir + "\\" + filename[:-3]+"undistorted.mp4")
            await self.send_status(websocket, f'Pose estimation complete')
            await self.send_status(websocket, f'Processing joint data...')
            new_joint_data = data_processing(joint_data,self.camera_params["mtx"],path_length)
            await self.send_status(websocket, f'Joint data processed')
            await self.send_status(websocket, f'Calculating gait measures...')
            measures = gait_measures(new_joint_data,"measures.xlsx")
            await self.send_status(websocket, f'Gait measures calculated')
            await self.send_status(websocket, f'success')
            await self.send_gait_data(websocket, measures)

            
            return filename
        except Exception as e:
            # Send error status
            await self.send_status(websocket, f'Video upload failed: {str(e)}')
            return None

    async def handle_client(self, websocket):
        """
        Main handler for each WebSocket client connection.
        """
        try:
            # Add client to connected clients
            self.connected_clients.add(websocket)
            
            # Send initial connection status
            await self.send_status(websocket, 'Connected to gait analysis server')
            
            async for message in websocket:
                try:
                    # Parse incoming message
                    parsed_message = json.loads(message)
                    print("Received message:", parsed_message)
                    # Handle different message types
                    if parsed_message.get('type') == 'gait_video_upload':
                        print("Received gait video")
                        # Handle video upload
                        video_data = parsed_message.get('video_data')
                        path_length = parsed_message.get('path_length')
                        if video_data and path_length:
                            await self.handle_gait_video_upload(websocket, video_data, path_length)
                    elif parsed_message.get('type') == 'checkerboard_video_upload':
                        # Handle video upload
                        video_data = parsed_message.get('video_data')
                        if video_data:
                            await self.handle_checkerboard_video_upload(websocket, video_data)

                    
                    # You can add more message type handlers here
                
                except json.JSONDecodeError:
                    await self.send_status(websocket, 'Invalid JSON format')
                except Exception as e:
                    await self.send_status(websocket, f'Error processing message: {str(e)}')
        
        except websockets.exceptions.ConnectionClosed:
            print(f"Client disconnected")
        finally:
            # Remove client from connected clients
            self.connected_clients.remove(websocket)

    async def start_server(self):
        """Start the WebSocket server."""
        server = await websockets.serve(
            self.handle_client, 
            self.host, 
            self.port,
            #extra_headers=[("Content-Security-Policy", "connect-src 'self' https://cdn.jsdelivr.net/pyodide/ ws://localhost:3000/")]
        )
        print(f"WebSocket server started on {self.host}:{self.port}")
        
        # Optional: Send periodic gait data to all connected clients
        # asyncio.create_task(self.periodic_gait_data_broadcast())
        
        await server.wait_closed()


def main():
    # Create and start the server
    server = VideoWebSocketServer(host='localhost', port=3000)
    
    # Run the server
    asyncio.run(server.start_server())

if __name__ == '__main__':
    main()
