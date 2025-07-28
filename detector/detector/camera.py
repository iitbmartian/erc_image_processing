import rclpy 
from rclpy.node import Node
 
from sensor_msgs.msg import Image
import cv2 
from ultralytics import YOLO 
 
 

import os 
import uuid 
import numpy as np 
from cv_bridge import CvBridge



os.makedirs("detections", exist_ok=True)
log_file = open("detection_log.txt", "w")


output_file = open("output_descriptions.txt", "w")
#client=genai.Client(api_key="AIzaSyDoUdAUvYv60ZQlC3Bxw4JoPw6tysVSWoI")
script_dir = os.path.dirname(os.path.realpath(__file__))
model_path = os.path.join(script_dir, "yoloe-2.pt")

# Load model
model = YOLO('/home/dhher/erc_ws/src/detector/detector/yoloe-2.pt')
 




class CameraSubscriber(Node):
    def __init__(self):
        super().__init__('camera_subscriber')
        self.bridge=CvBridge()
        self.subscription=self.create_subscription(
            Image,
            '/zedx/left/image_raw',
            self.detection,
            10)
        self.subscription
    def detection(self, video_stream):
        try:
            frame = self.bridge.imgmsg_to_cv2(video_stream, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"CV Bridge error: {e}")
            return
        results = model.predict(source=frame, stream=True, show=False, conf=0.25, imgsz=640)
        i=0
        for result in results:
            frame = result.orig_img
            h1,w1=frame.shape[:2]
            area_total=h1*w1
            boxes = result.boxes     # Boxes object
            for box in boxes:
                cls_id = int(box.cls[0])
                score = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                if(x1==0 or y1==0 or x2==0 or y2==0):
                    continue
                if(x1==w1 or x2==w1 or y1==h1 or y2==h1):
                    continue
            

                cropped=frame[y1:y2,x1:x2]
                h2,w2=cropped.shape[:2]
                
                area_cropped=h2*w2
                
                if(area_cropped > 0.6*area_total):
                    continue
                else:
                    i=i+1
                    target_size=(512,512)
                    cropped=cv2.resize(cropped,target_size,interpolation=cv2.INTER_CUBIC)
                    frame_id = result.path if hasattr(result, "path") else f"frame_{uuid.uuid4().hex[:8]}"
                    img_filename = f"{frame_id}+{i}.jpg"
                    cv2.imwrite(img_filename, cropped)
            cv2.imshow("YOLO11 Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        log_file.close()
        cap = None   
        cv2.destroyAllWindows()
def main(args=None):
    rclpy.init(args=args)
    camera_subscriber=CameraSubscriber()
    rclpy.spin(camera_subscriber)
    rclpy.shutdown() 
if __name__ == '__main__':
    main()
        
        

        