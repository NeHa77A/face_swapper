# # from fastapi import FastAPI, UploadFile, File, WebSocket
# # from fastapi.staticfiles import StaticFiles
# # from fastapi.responses import HTMLResponse
# # import cv2
# # import numpy as np
# # import base64
# # import insightface
# # from insightface.app import FaceAnalysis
# # import uvicorn
# # from pathlib import Path

# # app = FastAPI()
# # app.mount("/static", StaticFiles(directory="static"), name="static")

# # # Initialize face analysis and swapper
# # face_analyzer = FaceAnalysis(name='buffalo_l')
# # face_analyzer.prepare(ctx_id=-1, det_size=(320, 320))  # CPU mode
# # swapper = insightface.model_zoo.get_model('inswapper_128_fp16.onnx')

# # # Store uploaded source faces
# # source_faces = {}

# # def process_frame(frame_data, source_face):
# #     try:
# #         # Decode frame
# #         frame_bytes = base64.b64decode(frame_data.split(',')[1])
# #         nparr = np.frombuffer(frame_bytes, np.uint8)
# #         frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
# #         # Detect faces
# #         target_faces = face_analyzer.get(frame)
        
# #         if not target_faces:
# #             return None
            
# #         # Process frame
# #         result = frame.copy()
# #         for face in target_faces:
# #             result = swapper.get(result, face, source_face, paste_back=True)
        
# #         # Encode result
# #         _, buffer = cv2.imencode('.jpg', result)
# #         return base64.b64encode(buffer).decode('utf-8')
# #     except Exception as e:
# #         print(f"Error processing frame: {str(e)}")
# #         return None

# # @app.post("/upload_source")
# # async def upload_source(file: UploadFile = File(...)):
# #     try:
# #         contents = await file.read()
# #         nparr = np.frombuffer(contents, np.uint8)
# #         img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
# #         faces = face_analyzer.get(img)
# #         if len(faces) == 0:
# #             return {"error": "No face detected in the source image"}
        
# #         source_face_id = str(len(source_faces))
# #         source_faces[source_face_id] = faces[0]
        
# #         return {"face_id": source_face_id}
# #     except Exception as e:
# #         print(f"Upload error: {str(e)}")
# #         return {"error": "Failed to process the image"}

# # @app.websocket("/ws")
# # async def websocket_endpoint(websocket: WebSocket):
# #     await websocket.accept()
# #     try:
# #         while True:
# #             data = await websocket.receive_json()
# #             frame_data = data["frame"]
# #             face_id = data["face_id"]
            
# #             if face_id not in source_faces:
# #                 await websocket.send_json({"error": "Invalid face ID"})
# #                 continue
            
# #             result = process_frame(frame_data, source_faces[face_id])
# #             if result:
# #                 await websocket.send_json({"frame": f"data:image/jpeg;base64,{result}"})
            
# #     except Exception as e:
# #         print(f"WebSocket error: {str(e)}")
# #     finally:
# #         await websocket.close()

# # @app.get("/")
# # async def get_index():
# #     return HTMLResponse(Path("static/new_cam.html").read_text())

# # if __name__ == "__main__":
# #     uvicorn.run(app, host="0.0.0.0", port=8000)

# from fastapi import FastAPI, UploadFile, File, WebSocket
# from fastapi.staticfiles import StaticFiles
# from fastapi.responses import HTMLResponse
# import cv2
# import numpy as np
# import base64
# import insightface
# from insightface.app import FaceAnalysis
# import uvicorn
# from pathlib import Path

# app = FastAPI()
# app.mount("/static", StaticFiles(directory="static"), name="static")

# # Initialize face analysis and swapper (ensuring model is downloaded)
# # Initialize with specific model
# face_analyzer = FaceAnalysis(name='buffalo_l')
# face_analyzer.prepare(ctx_id=0, det_size=(320, 320))

# # Explicitly download and load the swapper model
# model_file = 'inswapper_128_fp16.onnx'
# swapper = insightface.model_zoo.get_model(model_file,
#                                          download=True,
#                                          download_zip=True)

# # Store uploaded source faces
# source_faces = {}

# def process_frame(frame_data, source_face):
#     try:
#         # Decode base64 frame
#         img_bytes = base64.b64decode(frame_data.split(',')[1])
#         nparr = np.frombuffer(img_bytes, np.uint8)
#         frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
#         if frame is None:
#             print("Failed to decode frame")
#             return None
            
#         # Get frame dimensions
#         height, width = frame.shape[:2]
#         if height == 0 or width == 0:
#             print("Invalid frame dimensions")
#             return None

#         # Detect faces in frame
#         faces = face_analyzer.get(frame)
        
#         if not faces:
#             print("No faces detected in frame")
#             return frame
            
#         # Process each detected face
#         result = frame.copy()
#         for det_face in faces:
#             try:
#                 result = swapper.get(result, det_face, source_face, paste_back=True)
#             except Exception as e:
#                 print(f"Error swapping face: {str(e)}")
#                 continue

#         # Encode processed frame
#         _, buffer = cv2.imencode('.jpg', result, [cv2.IMWRITE_JPEG_QUALITY, 85])
#         return base64.b64encode(buffer).decode('utf-8')
        
#     except Exception as e:
#         print(f"Error in process_frame: {str(e)}")
#         return None

# @app.post("/upload_source")
# async def upload_source(file: UploadFile = File(...)):
#     try:
#         contents = await file.read()
#         nparr = np.frombuffer(contents, np.uint8)
#         img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
#         if img is None:
#             return {"error": "Failed to decode image"}

#         # Detect faces in source image
#         faces = face_analyzer.get(img)
#         if not faces:
#             return {"error": "No face detected in source image"}
        
#         # Store the first detected face
#         source_face_id = str(len(source_faces))
#         source_faces[source_face_id] = faces[0]
#         print(f"Source face stored with ID: {source_face_id}")
        
#         return {"face_id": source_face_id}
        
#     except Exception as e:
#         print(f"Error in upload_source: {str(e)}")
#         return {"error": str(e)}

# @app.websocket("/ws")
# async def websocket_endpoint(websocket: WebSocket):
#     await websocket.accept()
    
#     try:
#         while True:
#             data = await websocket.receive_json()
            
#             if "frame" not in data or "face_id" not in data:
#                 await websocket.send_json({"error": "Invalid message format"})
#                 continue
                
#             face_id = data["face_id"]
#             if face_id not in source_faces:
#                 await websocket.send_json({"error": "Invalid face ID"})
#                 continue
                
#             result = process_frame(data["frame"], source_faces[face_id])
#             if result:
#                 await websocket.send_json({"frame": f"data:image/jpeg;base64,{result}"})
#             else:
#                 await websocket.send_json({"error": "Frame processing failed"})
            
#     except Exception as e:
#         print(f"WebSocket error: {str(e)}")
#     finally:
#         await websocket.close()

# @app.get("/")
# async def get_index():
#     return HTMLResponse(Path("static/new_cam.html").read_text())

# if __name__ == "__main__":
#     print("Starting server...")
#     print("Face analyzer and swapper initialized")
#     uvicorn.run(app, host="0.0.0.0", port=8000)


from fastapi import FastAPI, UploadFile, File, WebSocket
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import cv2
import numpy as np
import base64
import insightface
from insightface.app import FaceAnalysis
import uvicorn
from pathlib import Path
import asyncio
import time

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize face analysis and swapper
face_analyzer = FaceAnalysis(name='buffalo_l')
face_analyzer.prepare(ctx_id=0, det_size=(320, 320))
swapper = insightface.model_zoo.get_model('inswapper_128_fp16.onnx', download=True)

# Store uploaded source faces
source_faces = {}

# FPS control
MAX_FPS = 24
MIN_FRAME_TIME = 1.0 / MAX_FPS

def process_frame(frame_data, source_face):
    try:
        start_time = time.time()
        
        # Decode frame
        frame_bytes = base64.b64decode(frame_data.split(',')[1])
        nparr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return None
            
        # Detect faces
        faces = face_analyzer.get(frame)
        
        if not faces:
            return frame
            
        # Process frame
        result = frame.copy()
        for face in faces:
            result = swapper.get(result, face, source_face, paste_back=True)

        # Encode result with quality based on performance
        process_time = time.time() - start_time
        quality = max(60, min(95, int(100 - (process_time * 100))))  # Adjust quality based on processing time
        _, buffer = cv2.imencode('.jpg', result, [cv2.IMWRITE_JPEG_QUALITY, quality])
        
        return base64.b64encode(buffer).decode('utf-8')
        
    except Exception as e:
        print(f"Error in process_frame: {str(e)}")
        return None

@app.post("/upload_source")
async def upload_source(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return {"error": "Failed to decode image"}
            
        # Resize large images
        max_size = 640
        h, w = img.shape[:2]
        if h > max_size or w > max_size:
            scale = max_size / max(h, w)
            img = cv2.resize(img, (0, 0), fx=scale, fy=scale)
        
        faces = face_analyzer.get(img)
        if not faces:
            return {"error": "No face detected in source image"}
        
        source_face_id = str(len(source_faces))
        source_faces[source_face_id] = faces[0]
        print(f"Source face stored with ID: {source_face_id}")
        
        return {"face_id": source_face_id}
        
    except Exception as e:
        print(f"Error in upload_source: {str(e)}")
        return {"error": str(e)}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    last_frame_time = time.time()
    try:
        while True:
            # FPS control
            current_time = time.time()
            elapsed = current_time - last_frame_time
            if elapsed < MIN_FRAME_TIME:
                await asyncio.sleep(MIN_FRAME_TIME - elapsed)
                continue
                
            data = await websocket.receive_json()
            face_id = data.get("face_id")
            frame_data = data.get("frame")
            timestamp = data.get("timestamp", 0)
            
            if not all([face_id, frame_data]):
                await websocket.send_json({"error": "Invalid message format"})
                continue
            
            if face_id not in source_faces:
                await websocket.send_json({"error": "Invalid face ID"})
                continue
            
            # Process frame and measure time
            process_start = time.time()
            result = process_frame(frame_data, source_faces[face_id])
            process_time = time.time() - process_start
            
            if result:
                await websocket.send_json({
                    "frame": f"data:image/jpeg;base64,{result}",
                    "timestamp": timestamp,
                    "processTime": process_time * 1000  # Convert to milliseconds
                })
                last_frame_time = current_time
            else:
                await websocket.send_json({"error": "Frame processing failed"})
            
    except Exception as e:
        print(f"WebSocket error: {str(e)}")
    finally:
        await websocket.close()

@app.get("/")
async def get_index():
    return HTMLResponse(Path("static/new_cam.html").read_text())

if __name__ == "__main__":
    print("Starting server...")
    print(f"Face analyzer initialized with max FPS: {MAX_FPS}")
    uvicorn.run(app, host="0.0.0.0", port=8000)