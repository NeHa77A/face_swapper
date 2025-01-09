import cv2
import numpy as np
from keras_facenet import FaceNet
from mtcnn import MTCNN
import matplotlib.pyplot as plt
from gfpgan import GFPGANer
import os

def restore_image(image):
    """Restore image using GFPGAN"""
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

def process_and_restore_images(image1_path, image2_path, output_path, restored_output_path):
    # Initialize FaceNet and MTCNN
    embedder = FaceNet()
    detector = MTCNN()
    
    try:
        # Load and process first image (target)
        image1 = cv2.imread(image1_path)
        if image1 is None:
            raise ValueError("Could not read first image")
        image1_rgb = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
        
        # Load and process second image (source)
        image2 = cv2.imread(image2_path)
        if image2 is None:
            raise ValueError("Could not read second image")
        image2_rgb = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
        
        # Detect faces in both images
        faces1 = detector.detect_faces(image1_rgb)
        faces2 = detector.detect_faces(image2_rgb)
        
        if not faces1 or not faces2:
            raise ValueError("Could not detect faces in one or both images")
            
        # Get face boxes
        box1 = faces1[0]['box']
        box2 = faces2[0]['box']
        
        # Extract and preprocess faces
        x1, y1, w1, h1 = box1
        face1 = image1_rgb[y1:y1+h1, x1:x1+w1]
        face1 = cv2.resize(face1, (160, 160))
        face1 = np.expand_dims(face1, axis=0)
        
        x2, y2, w2, h2 = box2
        face2 = image2_rgb[y2:y2+h2, x2:x2+w2]
        face2 = cv2.resize(face2, (160, 160))
        face2 = np.expand_dims(face2, axis=0)
        
        # Get embeddings
        embedding1 = embedder.embeddings(face1)
        embedding2 = embedder.embeddings(face2)
        
        # Create modified embedding (replace first image's embedding with second's)
        modified_embedding = embedding2
        
        # Generate new face using the modified embedding
        reconstructed_face = cv2.resize(face2[0], (w1, h1))
        
        # Create output image (copy of first image)
        output_image = image1.copy()
        
        # Place the reconstructed face in the output image
        output_image[y1:y1+h1, x1:x1+w1] = cv2.cvtColor(reconstructed_face, cv2.COLOR_RGB2BGR)
        
        # Save the initial modified image
        cv2.imwrite(output_path, output_image)
        
        # Restore the modified image using GFPGAN
        print("Restoring image with GFPGAN...")
        restored_image = restore_image(output_image)
        
        # Save the restored image
        cv2.imwrite(restored_output_path, restored_image)
        
        # Display all results
        plt.figure(figsize=(20, 5))
        
        # Original image
        plt.subplot(1, 3, 1)
        plt.imshow(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))
        plt.title('Original Image')
        plt.axis('off')
        
        # Modified image (before restoration)
        plt.subplot(1, 3, 2)
        plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
        plt.title('Modified Image (Before Restoration)')
        plt.axis('off')
        
        # Restored image
        plt.subplot(1, 3, 3)
        plt.imshow(cv2.cvtColor(restored_image, cv2.COLOR_BGR2RGB))
        plt.title('Restored Image')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        return output_path, restored_output_path
        
    except Exception as e:
        print(f"Error during processing: {str(e)}")
        raise

def check_and_create_output_dir(directory):
    """Create output directory if it doesn't exist"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

# Usage
if __name__ == "__main__":
    # Set up paths
    image1_path = 'ajay.jpg'    # Target image to modify
    image2_path = 'khadus.jpg'  # Source image for face embedding
    
    # Create output directory
    output_dir = 'output'
    check_and_create_output_dir(output_dir)
    
    # Set output paths
    output_path = os.path.join(output_dir, 'embedded_face.jpg')
    restored_output_path = os.path.join(output_dir, 'restored_face.jpg')
    
    try:
        # Process and restore images
        modified_path, restored_path = process_and_restore_images(
            image1_path, 
            image2_path, 
            output_path, 
            restored_output_path
        )
        print(f"Modified image saved to: {modified_path}")
        print(f"Restored image saved to: {restored_path}")
    except Exception as e:
        print(f"Failed to process images: {str(e)}")