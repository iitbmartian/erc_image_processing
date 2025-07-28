from google import genai 
from PIL import Image
import io 
import uuid
import os
import cv2
import glob
from dotenv import  load_dotenv

load_dotenv()
api_key=os.getenv("API_KEY")

# Create output directory
os.makedirs("detections", exist_ok=True)
log_file = open("detection_log.txt", "w")



client=genai.Client(api_key=api_key)
def read_images_from_directory_cv2(directory_path):
     
    images = []
    # Define common image extensions
    image_extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff')

    for ext in image_extensions:
        # Construct the full path pattern for glob
        full_path_pattern = os.path.join(directory_path, ext)
        # Find all files matching the pattern
        for filename in glob.glob(full_path_pattern):
            img = cv2.imread(filename)

            if img is not None:
                images.append(img)
    return images
images=read_images_from_directory_cv2(".")
i=0
for image in images:
    i=i+1
    target_size=(512,512)
    cropped=cv2.resize(image,target_size,interpolation=cv2.INTER_CUBIC)
    
    _,buffer=cv2.imencode('.jpg',cropped)

    img=Image.open(io.BytesIO(buffer.tobytes()))
    response=client.models.generate_content(
        model="gemini-2.0-flash-lite",
        contents=[img,
                    "What are the objects in the image? Descibe them please give the objects and description in this format"
                    "If you have seen a similar or same object ignore it but type IGNORING"
                    "Object:-\n"
                    "Descriptions:-\n"
                    "Likelihood of being present on mars:-\n"
                    "Try to keep the description to 2-3 sentences max"
                    "For likelihood of being present on mars give me a percentage"]
        
    )
    description=response.text.strip()
    print(f'FOUND AT FRAME NUMBER{i}')
    log_file.write(f"FRAME NUMBER{i} object is:- \n ")
    log_file.write(description + "\n\n")
    log_file.flush() 
    cv2.imshow("IMAGE",image)
