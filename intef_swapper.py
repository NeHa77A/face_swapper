# import streamlit as st
# import cv2
# import numpy as np
# import insightface
# from insightface.app import FaceAnalysis
# from gfpgan import GFPGANer
# import tempfile
# from pathlib import Path

# def load_image(uploaded_file):
#     if uploaded_file is not None:
#         file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
#         img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
#         return img
#     return None

# def swap_faces(source_image, target_image):
#     app = FaceAnalysis(name='buffalo_l')
#     app.prepare(ctx_id=-1, det_size=(640, 640))
    
#     swapper = insightface.model_zoo.get_model('inswapper_128_fp16.onnx', download=True, download_zip=True)
    
#     source_faces = app.get(source_image)
#     if len(source_faces) == 0:
#         st.error("No faces detected in the source image!")
#         return None
        
#     target_faces = app.get(target_image)
#     if len(target_faces) == 0:
#         st.error("No faces detected in the target image!")
#         return None

#     source_face = source_faces[0]
#     result = target_image.copy()
#     for face in target_faces:
#         result = swapper.get(result, face, source_face, paste_back=True)
    
#     return result

# def restore_image(image):
#     gfpgan = GFPGANer(
#         model_path='GFPGANv1.4.pth',
#         upscale=2,
#         arch='clean',
#         channel_multiplier=2,
#         bg_upsampler=None
#     )
    
#     _, _, restored_image = gfpgan.enhance(image, has_aligned=False, only_center_face=False, paste_back=True)
#     return restored_image

# def main():
#     st.title("Face Swap App")
    
#     col1, col2 = st.columns(2)
    
#     with col1:
#         st.subheader("Source Image (Face to use)")
#         source_file = st.file_uploader("Upload source image", type=['jpg', 'jpeg', 'png'])
#         if source_file:
#             source_image = load_image(source_file)
#             st.image(source_file, caption="Source Image")
        
#     with col2:
#         st.subheader("Target Image (Face to replace)")
#         target_file = st.file_uploader("Upload target image", type=['jpg', 'jpeg', 'png'])
#         if target_file:
#             target_image = load_image(target_file)
#             st.image(target_file, caption="Target Image")
    
#     enhance = st.checkbox("Apply GFPGAN enhancement")
    
#     if st.button("Swap Faces") and source_file and target_file:
#         with st.spinner("Processing..."):
#             source_image = load_image(source_file)
#             target_image = load_image(target_file)
            
#             result = swap_faces(source_image, target_image)
            
#             if result is not None:
#                 if enhance:
#                     result = restore_image(result)
                
#                 # Convert BGR to RGB for display
#                 result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
#                 st.image(result_rgb, caption="Result")
                
#                 # Save button
#                 result_bgr = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)
#                 _, encoded_img = cv2.imencode('.png', result_bgr)
#                 st.download_button(
#                     label="Download result",
#                     data=encoded_img.tobytes(),
#                     file_name="swapped_face.png",
#                     mime="image/png"
#                 )

# if __name__ == '__main__':
#     main()


import streamlit as st
from streamlit_option_menu import option_menu
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from gfpgan import GFPGANer

st.set_page_config(
    page_title="AI Face Swap Studio",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    .uploadedFile {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    .stButton>button {
        background: linear-gradient(45deg, #4A148C, #311B92);
        color: white;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        border: none;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 8px rgba(0, 0, 0, 0.4);
        background: linear-gradient(45deg, #311B92, #4A148C);
    }
    .css-1d391kg {
        background-color: #1E1E1E;
    }
    .stTextInput>div>div>input {
        background-color: #2D2D2D;
        color: white;
    }
    .stMarkdown {
        color: #FAFAFA;
    }
    .css-1v0mbdj.e115fcil1 {
        background-color: #1E1E1E;
        border-color: #333;
    }
    </style>
""", unsafe_allow_html=True)

def load_image(uploaded_file):
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        return img
    return None

def swap_faces(source_image, target_image):
    app = FaceAnalysis(name='buffalo_l')
    app.prepare(ctx_id=-1, det_size=(640, 640))
    
    with st.spinner("📥 Loading face swap model..."):
        swapper = insightface.model_zoo.get_model('inswapper_128_fp16.onnx', download=True, download_zip=True)
    
    source_faces = app.get(source_image)
    if len(source_faces) == 0:
        st.error("❌ No faces detected in the source image!")
        return None
        
    target_faces = app.get(target_image)
    if len(target_faces) == 0:
        st.error("❌ No faces detected in the target image!")
        return None

    source_face = source_faces[0]
    result = target_image.copy()
    for face in target_faces:
        result = swapper.get(result, face, source_face, paste_back=True)
    
    return result

def restore_image(image):
    with st.spinner("🔄 Enhancing image quality..."):
        gfpgan = GFPGANer(
            model_path='GFPGANv1.4.pth',
            upscale=2,
            arch='clean',
            channel_multiplier=2,
            bg_upsampler=None
        )
        
        _, _, restored_image = gfpgan.enhance(image, has_aligned=False, only_center_face=False, paste_back=True)
    return restored_image

def main():
    # Sidebar
    with st.sidebar:
        selected = option_menu(
            menu_title="Navigation",
            options=["Home", "About"],
            icons=["house", "info-circle"],
            menu_icon="cast",
            default_index=0,
        )
    
    if selected == "Home":
        st.title("🎭 AI Face Swap Studio")
        st.markdown("### Transform your photos with advanced AI face swapping technology")
        
        # Main content
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📸 Source Image")
            source_file = st.file_uploader(
                "Upload the face you want to use",
                type=['jpg', 'jpeg', 'png'],
                help="This face will be swapped onto the target image"
            )
            if source_file:
                st.image(source_file, caption="Source Face", use_column_width=True)
        
        with col2:
            st.markdown("### 🎯 Target Image")
            target_file = st.file_uploader(
                "Upload the image to replace face in",
                type=['jpg', 'jpeg', 'png'],
                help="The face in this image will be replaced"
            )
            if target_file:
                st.image(target_file, caption="Target Image", use_column_width=True)
        
        # Settings
        st.markdown("### ⚙️ Settings")
        col3, col4 = st.columns(2)
        with col3:
            enhance = st.checkbox("✨ Enable AI Enhancement", 
                                help="Apply GFPGAN to improve the final result")
        
        # Process button
        if source_file and target_file:
            if st.button("🔄 Process Face Swap", use_container_width=True):
                try:
                    with st.spinner("🎨 Performing face swap..."):
                        source_image = load_image(source_file)
                        target_image = load_image(target_file)
                        
                        result = swap_faces(source_image, target_image)
                        
                        if result is not None:
                            if enhance:
                                result = restore_image(result)
                            
                            # Display result
                            st.markdown("### ✨ Result")
                            result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
                            st.image(result_rgb, caption="Final Result", use_column_width=True)
                            
                            # Download button
                            result_bgr = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)
                            _, encoded_img = cv2.imencode('.png', result_bgr)
                            st.download_button(
                                label="💾 Download Result",
                                data=encoded_img.tobytes(),
                                file_name="face_swap_result.png",
                                mime="image/png",
                                use_container_width=True
                            )
                except Exception as e:
                    st.error(f"❌ An error occurred: {str(e)}")
    
    elif selected == "About":
        st.title("About AI Face Swap Studio")
        st.markdown("""
        ### 🚀 Features
        - Advanced AI-powered face swapping
        - GFPGAN enhancement for better quality
        - User-friendly interface
        - Instant download of results
        
        ### 🔒 Privacy
        Your uploads are processed locally and are not stored permanently.
        
        ### 📝 How to Use
        1. Upload a source image containing the face you want to use
        2. Upload a target image where you want to replace the face
        3. Enable AI enhancement if desired
        4. Click process and download your result
        """)

if __name__ == '__main__':
    main()