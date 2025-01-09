from fastapi import FastAPI, UploadFile, File, WebSocket, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import base64
import insightface
from insightface.app import FaceAnalysis
import uvicorn
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Enable CORS - Important for Chrome extension
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Initialize face analysis
try:
    logger.info("Initializing face analyzer...")
    face_analyzer = FaceAnalysis(name='buffalo_l')
    face_analyzer.prepare(ctx_id=0, det_size=(320, 320))
    
    logger.info("Loading face swapper model...")
    swapper = insightface.model_zoo.get_model('inswapper_128_fp16.onnx', download=True)
    
    logger.info("Models loaded successfully")
except Exception as e:
    logger.error(f"Error initializing models: {str(e)}")
    raise

# Store uploaded source faces
source_faces = {}

def process_frame(frame_data, source_face):
    try:
        # Decode frame
        if ',' in frame_data:
            frame_data = frame_data.split(',')[1]
        
        frame_bytes = base64.b64decode(frame_data)
        nparr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            logger.error("Failed to decode frame")
            return None
            
        # Detect faces
        faces = face_analyzer.get(frame)
        
        if not faces:
            logger.info("No faces detected in frame")
            _, buffer = cv2.imencode('.jpg', frame)
            return f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"
            
        # Process frame
        result = frame.copy()
        for face in faces:
            result = swapper.get(result, face, source_face, paste_back=True)
            
        # Encode result
        _, buffer = cv2.imencode('.jpg', result, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"
        
    except Exception as e:
        logger.error(f"Error in process_frame: {str(e)}")
        return None

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket connection established")
    
    try:
        while True:
            data = await websocket.receive_json()
            
            face_id = data.get("face_id")
            frame_data = data.get("frame")
            
            if not all([face_id, frame_data]):
                await websocket.send_json({"error": "Invalid message format"})
                continue
                
            if face_id not in source_faces:
                await websocket.send_json({"error": "Invalid face ID"})
                continue
                
            result = process_frame(frame_data, source_faces[face_id])
            
            if result:
                await websocket.send_json({
                    "frame": result,
                    "timestamp": data.get("timestamp", 0)
                })
            else:
                await websocket.send_json({"error": "Frame processing failed"})
                
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
    finally:
        logger.info("WebSocket connection closed")
        await websocket.close()

@app.post("/upload_source")
async def upload_source(file: UploadFile = File(...)):
    try:
        logger.info("Receiving source face upload")
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Failed to decode image")
            
        # Resize large images
        max_size = 640
        h, w = img.shape[:2]
        if h > max_size or w > max_size:
            scale = max_size / max(h, w)
            img = cv2.resize(img, (0, 0), fx=scale, fy=scale)
            
        faces = face_analyzer.get(img)
        if not faces:
            raise HTTPException(status_code=400, detail="No face detected in source image")
            
        face_id = str(len(source_faces))
        source_faces[face_id] = faces[0]
        logger.info(f"Source face stored with ID: {face_id}")
        
        return {"face_id": face_id}
        
    except Exception as e:
        logger.error(f"Error in upload_source: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "online"}

if __name__ == "__main__":
    logger.info("Starting server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)