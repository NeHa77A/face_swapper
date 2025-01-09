# import os
# import cv2
# import numpy as np
# import insightface
# from insightface.app import FaceAnalysis
# import logging
# import time
# import json
# from typing import List, Tuple, Dict
# from pathlib import Path
# import asyncio
# from concurrent.futures import ThreadPoolExecutor

# from fastapi import FastAPI, WebSocket, UploadFile, File, HTTPException, WebSocketDisconnect
# from fastapi.staticfiles import StaticFiles
# from fastapi.responses import HTMLResponse, JSONResponse
# from fastapi.middleware.cors import CORSMiddleware

# # Setup logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # Initialize FastAPI app
# app = FastAPI()

# # Enable CORS
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Create static directory if it doesn't exist
# static_dir = Path("static")
# static_dir.mkdir(exist_ok=True)

# # Mount static files directory
# app.mount("/static", StaticFiles(directory="static"), name="static")

# # Create thread pool for CPU-intensive operations
# thread_pool = ThreadPoolExecutor(max_workers=4)

# class ConnectionManager:
#     def __init__(self):
#         self.active_connections: List[WebSocket] = []

#     async def connect(self, websocket: WebSocket):
#         await websocket.accept()
#         self.active_connections.append(websocket)
#         logger.info(f"Client connected. Total connections: {len(self.active_connections)}")

#     def disconnect(self, websocket: WebSocket):
#         self.active_connections.remove(websocket)
#         logger.info(f"Client disconnected. Total connections: {len(self.active_connections)}")

#     async def broadcast(self, message: str):
#         for connection in self.active_connections:
#             await connection.send_text(message)

# class FaceSwapProcessor:
#     def __init__(self):
#         # Initialize face analysis
#         self.app = FaceAnalysis(name='buffalo_l')
#         self.app.prepare(ctx_id=-1, det_size=(640, 640))
        
#         # Initialize face swapper
#         model_path = os.path.join(os.path.dirname(__file__), 'inswapper_128_fp16.onnx')
#         self.swapper = insightface.model_zoo.get_model(model_path)
        
#         self.source_faces: Dict[str, np.ndarray] = {}
#         self.current_source_id: str = None
#         self.processing_stats = {
#             'total_frames': 0,
#             'faces_detected': 0,
#             'processing_times': []
#         }

#     async def add_source_face(self, image_data: bytes, source_id: str) -> bool:
#         try:
#             # Convert bytes to numpy array
#             nparr = np.frombuffer(image_data, np.uint8)
#             image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
#             # Detect face in source image
#             faces = self.app.get(image)
#             if len(faces) > 0:
#                 self.source_faces[source_id] = faces[0]
#                 return True
#             return False
#         except Exception as e:
#             logger.error(f"Error adding source face: {e}")
#             return False

#     async def process_frame(self, frame: np.ndarray) -> np.ndarray:
#         if self.current_source_id not in self.source_faces:
#             return frame

#         try:
#             source_face = self.source_faces[self.current_source_id]
#             target_faces = self.app.get(frame)
            
#             if len(target_faces) == 0:
#                 return frame

#             result = frame.copy()
#             for face in target_faces:
#                 result = self.swapper.get(result, face, source_face, paste_back=True)
            
#             return result
#         except Exception as e:
#             logger.error(f"Error processing frame: {e}")
#             return frame

#     async def process_frame_with_stats(self, frame: np.ndarray) -> Tuple[np.ndarray, dict]:
#         start_time = time.time()
        
#         result = await self.process_frame(frame)
        
#         # Update statistics
#         self.processing_stats['total_frames'] += 1
#         faces = self.app.get(frame)
#         self.processing_stats['faces_detected'] += len(faces)
#         processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds
#         self.processing_stats['processing_times'].append(processing_time)
        
#         # Keep only last 100 processing times for average calculation
#         if len(self.processing_stats['processing_times']) > 100:
#             self.processing_stats['processing_times'].pop(0)
        
#         stats = {
#             'processing_time': processing_time,
#             'faces_detected': len(faces),
#             'avg_processing_time': np.mean(self.processing_stats['processing_times'])
#         }
        
#         return result, stats

# # Initialize managers and processor
# manager = ConnectionManager()
# processor = FaceSwapProcessor()

# @app.get("/")
# async def read_root():
#     """Serve the main HTML page"""
#     html_path = static_dir / "new_cam.html"
#     if html_path.exists():
#         return HTMLResponse(html_path.read_text())
#     return {"message": "Welcome to Face Swap API"}

# @app.post("/upload-source")
# async def upload_source_face(file: UploadFile = File(...)):
#     """Handle source face image upload"""
#     try:
#         contents = await file.read()
#         source_id = file.filename.split('.')[0]
#         success = await processor.add_source_face(contents, source_id)
        
#         return {
#             "success": success,
#             "source_id": source_id,
#             "message": "Face added successfully" if success else "No face detected"
#         }
#     except Exception as e:
#         logger.error(f"Error uploading source face: {e}")
#         raise HTTPException(status_code=500, detail=str(e))

# @app.get("/stats")
# async def get_stats():
#     """Get current processing statistics"""
#     if not processor.processing_stats['processing_times']:
#         return JSONResponse(content={
#             "message": "No processing statistics available yet"
#         })
    
#     return {
#         "total_frames_processed": processor.processing_stats['total_frames'],
#         "total_faces_detected": processor.processing_stats['faces_detected'],
#         "average_processing_time": np.mean(processor.processing_stats['processing_times']),
#         "active_connections": len(manager.active_connections)
#     }

# @app.websocket("/ws")
# async def websocket_endpoint(websocket: WebSocket):
#     """WebSocket endpoint for real-time face swapping"""
#     await manager.connect(websocket)
#     try:
#         while True:
#             message = await websocket.receive()
            
#             if message.get("type") == "bytes":
#                 try:
#                     # Process frame in thread pool to avoid blocking
#                     frame_data = message.get("bytes")
#                     nparr = np.frombuffer(frame_data, np.uint8)
#                     frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    
#                     if frame is not None:
#                         # Process frame and get stats
#                         result, stats = await processor.process_frame_with_stats(frame)
                        
#                         # Encode and send processed frame
#                         _, buffer = cv2.imencode('.jpg', result, [cv2.IMWRITE_JPEG_QUALITY, 85])
#                         await websocket.send_bytes(buffer.tobytes())
                        
#                         # Send stats
#                         await websocket.send_json({
#                             "type": "stats",
#                             "data": stats
#                         })
#                 except Exception as e:
#                     logger.error(f"Error processing frame: {e}")
#                     await websocket.send_json({
#                         "type": "error",
#                         "message": "Error processing frame"
#                     })
            
#             elif message.get("type") == "text":
#                 try:
#                     data = json.loads(message.get("text"))
#                     if data.get("type") == "source_change":
#                         processor.current_source_id = data.get("sourceId")
#                         await websocket.send_json({
#                             "type": "status",
#                             "message": f"Source changed to {data.get('sourceId')}"
#                         })
                    
#                     elif data.get("type") == "get_stats":
#                         await websocket.send_json({
#                             "type": "stats",
#                             "data": processor.processing_stats
#                         })
#                 except json.JSONDecodeError:
#                     logger.error("Invalid JSON message received")
    
#     except WebSocketDisconnect:
#         manager.disconnect(websocket)
#     except Exception as e:
#         logger.error(f"WebSocket error: {e}")
#         manager.disconnect(websocket)

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)
from fastapi import FastAPI, UploadFile, File, WebSocket
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
import cv2
import numpy as np
import base64
import insightface
from insightface.app import FaceAnalysis
import uvicorn
from pathlib import Path
import asyncio
import json

app = FastAPI()

# Mount the static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize face analysis and swapper
face_analyzer = FaceAnalysis(name='buffalo_l')
face_analyzer.prepare(ctx_id=-1, det_size=(640, 640))
swapper = insightface.model_zoo.get_model('inswapper_128_fp16.onnx', download=True, download_zip=True)

# Store uploaded source faces
source_faces = {}

def process_frame(frame_data, source_face):
    # Decode base64 frame
    frame_bytes = base64.b64decode(frame_data.split(',')[1])
    nparr = np.frombuffer(frame_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Detect and swap faces
    target_faces = face_analyzer.get(frame)
    if len(target_faces) == 0:
        return frame
    
    result = frame.copy()
    for face in target_faces:
        result = swapper.get(result, face, source_face, paste_back=True)
    
    # Encode the result back to base64
    _, buffer = cv2.imencode('.jpg', result)
    return base64.b64encode(buffer).decode('utf-8')

@app.post("/upload_source")
async def upload_source(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    faces = face_analyzer.get(img)
    if len(faces) == 0:
        return JSONResponse({"error": "No face detected in the source image"}, status_code=400)
    
    source_face_id = str(len(source_faces))
    source_faces[source_face_id] = faces[0]
    
    return {"face_id": source_face_id}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    try:
        while True:
            data = await websocket.receive_json()
            frame_data = data["frame"]
            face_id = data["face_id"]
            
            if face_id not in source_faces:
                await websocket.send_json({"error": "Invalid face ID"})
                continue
            
            result = process_frame(frame_data, source_faces[face_id])
            await websocket.send_json({"frame": f"data:image/jpeg;base64,{result}"})
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        await websocket.close()

@app.get("/")
async def get_index():
    return HTMLResponse(Path("static/index.html").read_text())

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)