import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from gfpgan import GFPGANer

def swap_faces_live(source_face, frame, app, swapper):
    # Detect faces in the current frame
    target_faces = app.get(frame)
    if len(target_faces) == 0:
        return frame  # Return the original frame if no faces detected

    # Perform face swapping for each detected face in the frame
    result = frame.copy()
    for face in target_faces:
        result = swapper.get(result, face, source_face, paste_back=True)
    return result

if __name__ == '__main__':
    # Load and preprocess source images
    source_images = {
        "1": cv2.imread("Saket_Srivastava_sir.png"),  # Replace with your source image paths
        "2": cv2.imread("ajay.jpg"),
    }

    # Initialize the FaceAnalysis app for detection
    app = FaceAnalysis(name='buffalo_l')
    app.prepare(ctx_id=-1, det_size=(640, 640))  # Use GPU if available; ctx_id=-1 for CPU.

    # Load the face swapping model
    swapper = insightface.model_zoo.get_model('inswapper_128_fp16.onnx', download=True, download_zip=True)

    # Extract face embeddings for each source image
    source_faces = {}
    for key, image in source_images.items():
        faces = app.get(image)
        if len(faces) > 0:
            source_faces[key] = faces[0]  # Save the first detected face

    if not source_faces:
        raise ValueError("No valid faces detected in any source images!")

    # Start webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise ValueError("Unable to access the webcam.")

    print("Press the corresponding number to select a source face (1/2). Press 'q' to quit.")
    selected_face_key = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame. Exiting...")
            break

        # Swap faces if a source face is selected
        if selected_face_key in source_faces:
            frame = swap_faces_live(source_faces[selected_face_key], frame, app, swapper)

        # Display the live video feed
        cv2.imshow("Live Face Swap", frame)

        # Check for key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # Quit
            break
        elif chr(key) in source_faces:  # Change source face
            selected_face_key = chr(key)
            print(f"Selected face: {selected_face_key}")

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
