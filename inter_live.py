# import streamlit as st
# import cv2
# import numpy as np
# import insightface
# from insightface.app import FaceAnalysis
# import tempfile
# import os
# from PIL import Image

# def load_image(image_file):
#     if image_file is not None:
#         image = Image.open(image_file)
#         return np.array(image)
#     return None

# def process_frame(frame, source_face, app, swapper):
#     # Detect faces in the current frame
#     target_faces = app.get(frame)
#     if len(target_faces) == 0:
#         return frame  # Return original frame if no faces detected
    
#     # Perform face swapping for each detected face
#     result = frame.copy()
#     for face in target_faces:
#         result = swapper.get(result, face, source_face, paste_back=True)
#     return result

# def main():
#     st.title("Real-Time Face Swap Application")
    
#     # Initialize face analysis and swapper in session state
#     if 'app' not in st.session_state:
#         st.session_state.app = FaceAnalysis(name='buffalo_l')
#         st.session_state.app.prepare(ctx_id=-1, det_size=(640, 640))
        
#     if 'swapper' not in st.session_state:
#         st.session_state.swapper = insightface.model_zoo.get_model(
#             'inswapper_128_fp16.onnx',
#             download=True,
#             download_zip=True
#         )

#     # Sidebar for source image upload
#     st.sidebar.header("Upload Source Face")
#     source_image = st.sidebar.file_uploader("Choose a source face image", type=['jpg', 'jpeg', 'png'])
    
#     # Initialize source face
#     source_face = None
#     if source_image is not None:
#         source_img = load_image(source_image)
#         if source_img is not None:
#             faces = st.session_state.app.get(source_img)
#             if len(faces) > 0:
#                 source_face = faces[0]
#                 st.sidebar.image(source_img, caption="Source Face", use_column_width=True)
#             else:
#                 st.sidebar.error("No face detected in the source image!")

#     # Main area for target image processing
#     st.header("Target Image Processing")
#     option = st.radio("Choose input type:", ["Upload Image", "Use Webcam"])

#     if option == "Upload Image":
#         target_image = st.file_uploader("Choose a target image", type=['jpg', 'jpeg', 'png'])
#         if target_image is not None and source_face is not None:
#             target_img = load_image(target_image)
#             if target_img is not None:
#                 # Process the image
#                 result = process_frame(target_img, source_face, st.session_state.app, st.session_state.swapper)
                
#                 # Display original and processed images side by side
#                 col1, col2 = st.columns(2)
#                 with col1:
#                     st.image(target_img, caption="Original Image")
#                 with col2:
#                     st.image(result, caption="Processed Image")

#     else:  # Webcam option
#         if source_face is not None:
#             st.write("Webcam Feed (Click 'Start' to begin)")
#             run = st.checkbox('Start')
#             FRAME_WINDOW = st.image([])
#             camera = cv2.VideoCapture(0)

#             while run:
#                 _, frame = camera.read()
#                 if frame is not None:
#                     # Convert BGR to RGB
#                     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#                     # Process the frame
#                     result = process_frame(frame, source_face, st.session_state.app, st.session_state.swapper)
#                     FRAME_WINDOW.image(result)
#                 else:
#                     st.warning("Error accessing webcam feed!")
#                     break

#             if not run:
#                 camera.release()
#                 st.write("Webcam stopped")
#         else:
#             st.warning("Please upload a source face image first!")

#     # Footer
#     st.markdown("---")
#     st.markdown("### Instructions:")
#     st.markdown("""
#     1. Upload a source face image in the sidebar
#     2. Choose between uploading a target image or using your webcam
#     3. For uploaded images, the result will show immediately
#     4. For webcam, click 'Start' to begin face swapping
#     """)

# if __name__ == '__main__':
#     main()

### GAN is added
import streamlit as st
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
import tempfile
import os
from PIL import Image
from gfpgan import GFPGANer

def load_image(image_file):
    if image_file is not None:
        image = Image.open(image_file)
        return np.array(image)
    return None

def restore_image(image):
    # Initialize the GFPGAN model for restoration
    gfpgan = GFPGANer(
        model_path='GFPGANv1.4.pth',  # Model will be downloaded automatically
        upscale=2,  # Upscaling factor
        arch='clean',  # Use the clean architecture
        channel_multiplier=2,  # Channel multiplier for quality adjustment
        bg_upsampler=None  # Use default background upsampler
    )

    # Perform restoration
    _, _, restored_image = gfpgan.enhance(image, has_aligned=False, only_center_face=False, paste_back=True)
    return restored_image

def process_frame(frame, source_face, app, swapper, apply_restoration=False):
    # Detect faces in the current frame
    target_faces = app.get(frame)
    if len(target_faces) == 0:
        return frame  # Return original frame if no faces detected
    
    # Perform face swapping for each detected face
    result = frame.copy()
    for face in target_faces:
        result = swapper.get(result, face, source_face, paste_back=True)
    
    # Apply restoration if requested
    if apply_restoration:
        result = restore_image(result)
    
    return result

def main():
    st.title("Real-Time Face Swap Application with Image Restoration")
    
    # Initialize face analysis and swapper in session state
    if 'app' not in st.session_state:
        st.session_state.app = FaceAnalysis(name='buffalo_l')
        st.session_state.app.prepare(ctx_id=-1, det_size=(640, 640))
        
    if 'swapper' not in st.session_state:
        st.session_state.swapper = insightface.model_zoo.get_model(
            'inswapper_128_fp16.onnx',
            download=True,
            download_zip=True
        )

    # Sidebar for source image upload and settings
    st.sidebar.header("Settings")
    source_image = st.sidebar.file_uploader("Choose a source face image", type=['jpg', 'jpeg', 'png'])
    apply_restoration = st.sidebar.checkbox("Apply Image Restoration (GFPGAN)", value=False)
    
    if apply_restoration:
        st.sidebar.info("Image restoration will enhance face quality but may increase processing time.")
    
    # Initialize source face
    source_face = None
    if source_image is not None:
        source_img = load_image(source_image)
        if source_img is not None:
            faces = st.session_state.app.get(source_img)
            if len(faces) > 0:
                source_face = faces[0]
                st.sidebar.image(source_img, caption="Source Face", use_column_width=True)
            else:
                st.sidebar.error("No face detected in the source image!")

    # Main area for target image processing
    st.header("Target Image Processing")
    option = st.radio("Choose input type:", ["Upload Image", "Use Webcam"])

    if option == "Upload Image":
        target_image = st.file_uploader("Choose a target image", type=['jpg', 'jpeg', 'png'])
        if target_image is not None and source_face is not None:
            target_img = load_image(target_image)
            if target_img is not None:
                # Process the image
                with st.spinner("Processing image..."):
                    result = process_frame(
                        target_img, 
                        source_face, 
                        st.session_state.app, 
                        st.session_state.swapper,
                        apply_restoration
                    )
                
                # Display original and processed images side by side
                col1, col2 = st.columns(2)
                with col1:
                    st.image(target_img, caption="Original Image")
                with col2:
                    st.image(result, caption="Processed Image")

                # Add download button for processed image
                if result is not None:
                    # Convert numpy array to PIL Image
                    result_pil = Image.fromarray(result)
                    # Create a bytes buffer for the image
                    buf = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
                    result_pil.save(buf, format='PNG')
                    # Add download button
                    with open(buf.name, 'rb') as f:
                        st.download_button(
                            label="Download processed image",
                            data=f,
                            file_name="processed_image.png",
                            mime="image/png"
                        )

    else:  # Webcam option
        if source_face is not None:
            st.write("Webcam Feed (Click 'Start' to begin)")
            run = st.checkbox('Start')
            FRAME_WINDOW = st.image([])
            camera = cv2.VideoCapture(0)

            while run:
                _, frame = camera.read()
                if frame is not None:
                    # Convert BGR to RGB
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # Process the frame
                    result = process_frame(
                        frame, 
                        source_face, 
                        st.session_state.app, 
                        st.session_state.swapper,
                        apply_restoration
                    )
                    FRAME_WINDOW.image(result)
                else:
                    st.warning("Error accessing webcam feed!")
                    break

            if not run:
                camera.release()
                st.write("Webcam stopped")
        else:
            st.warning("Please upload a source face image first!")

    # Footer
    st.markdown("---")
    st.markdown("### Instructions:")
    st.markdown("""
    1. Upload a source face image in the sidebar
    2. Choose whether to enable image restoration (GFPGAN)
    3. Choose between uploading a target image or using your webcam
    4. For uploaded images, the result will show immediately
    5. For webcam, click 'Start' to begin face swapping
    6. Download the processed image if desired
    """)

if __name__ == '__main__':
    main()