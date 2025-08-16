  

import math
import rclpy 
from rclpy.node import Node
 
from sensor_msgs.msg import Image
import cv2 
from ultralytics import YOLO 
from geometry_msgs.msg import PoseWithCovarianceStamped
 
 

import os 
import uuid 
import numpy as np 
from cv_bridge import CvBridge
import time



os.makedirs("detections", exist_ok=True)
log_file = open("detection_log.txt", "w")


output_file = open("output_descriptions.txt", "w")

script_dir = os.path.dirname(os.path.realpath(__file__))
model_path = os.path.join(script_dir, "yoloe-2.pt")
 
model = YOLO('/home/dhher/erc_ws/src/detector/detector/yoloe-2.pt')

 


coords=set() 
class CameraSubscriber(Node):
    def __init__(self):
        super().__init__('camera_subscriber')
        self.bridge=CvBridge()
        self.subscription=self.create_subscription(
            Image,
            '/zedx/left/image_raw',
            self.detection,
            10)
        # self.subscriber_2=self.create_subscription(
        #     PoseWithCovarianceStamped,
        #     '/rtabmap/localization_pose',
        #     self.detection,
        #     10)
        self.subscription
        #self.subscriber_2
        self.i=0
        self.objects_clases_frequency={}
        self.masks_data=[]
        self.threshold_dice = 0.15
        self.threshold_iou = 0.15
        self.classes_detected=[]
        self.prev=[]
                
        
    def calc_dis(self, coords_1,coords_2):
        return math.sqrt(abs(coords_1[0]-coords_2[0])**2 + (coords_1[1]-coords_2[1])**2 + (coords_1[2]-coords_2[2])**2)
    def mask_dice(self,mask1,mask2):
         
        intersection = np.logical_and(mask1, mask2).sum()
        return 2* intersection / (mask1.sum() + mask2.sum())
    def mask_iou(mself,mask1, mask2):
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        return intersection / union if union != 0 else 0.0
    def detection(self, video_stream):
       
        try:
            frame = self.bridge.imgmsg_to_cv2(video_stream, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"CV Bridge error: {e}")
            return
         
        results = model.predict(source=frame, stream=True, show=False, conf=0.25, imgsz=640,verbose=False)
        # x = msg.pose.pose.position.x
        # y = msg.pose.pose.position.y
        # z = msg.pose.pose.position.z
        # t=(int(x),int(y),int(z))
        
        for result in results:

            
            frame = result.orig_img
            h1,w1=frame.shape[:2]
            area_total=h1*w1
            boxes = result.boxes
            boxes=list(boxes)   
            
            for box,mask in zip(boxes, result.masks):
                continuer = True
                 
                mask_array=mask.data
                if(len(self.masks_data)>9):
                    masks_checker=self.masks_data[-10:]
                    for i in masks_checker:
                        if(self.mask_dice(mask_array,i)>self.threshold_dice or self.mask_iou(mask_array,i)>self.threshold_iou):
                            continuer=False
                            break
                if(not continuer):
                    continue

                self.masks_data.append(mask_array)
                    
                 
                
                


                class_index=int(box.cls)
                class_name=result.names[class_index]
                # if(class_name=="desert" or class_name=="sky" or class_name=="drought" or class_name=="sand" or class_name=="fly"):
                #     continue
                # continuer=True
                # if(class_name in self.objects_clases_frequency):
                #     if(self.objects_clases_frequency[class_name]==True or class_name==self.prev_class):
                #         self.objects_clases_frequency[class_name] =False
                #         continuer=False
                # else:
                    #self.objects_clases_frequency[class_name]=True
                self.prev=self.classes_detected[-10:]
                if class_name in self.prev:
                    continue
                if(continuer==False):
                    continue

                    
                     

 
                 
                cls_id = int(box.cls[0])
                score = float(box.conf[0])
                if(score < 0.5):
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                if(x1==0 or y1==0 or x2==0 or y2==0):
                    continue
                if(x1==w1 or x2==w1 or y1==h1 or y2==h1):
                    continue
            
                # box1_coordinates=[x1,y1,x2,y2]
                # px1,py1,px2,py2=map(int,prev_box.xyxy[0])
                # box2_coordinates=[px1,py1,px2,py2]
                # iou=self.calculate_iou(box1_coordinates,box2_coordinates)
 

                cropped=frame[y1:y2,x1:x2]
                h2,w2=cropped.shape[:2]
                
                area_cropped=h2*w2
                
                if(area_cropped > 0.6*area_total):
                    continue
                else:
                    
                    target_size=(512,512)
                    cropped=cv2.resize(cropped,target_size,interpolation=cv2.INTER_CUBIC)
                    frame_id = result.path if hasattr(result, "path") else f"frame_{uuid.uuid4().hex[:8]}"
                    img_filename = f"{frame_id}+{self.i}.jpg"
                    unique_id = f"{int(time.time() * 1000)}_{uuid.uuid4().hex[:4]}"
                    img_filename = f"{unique_id}_{self.i}.jpg"
                    cv2.imwrite(img_filename, cropped)
                    print(f'{self.i}+{class_name}+{score}')
                    print("BREAK")
                    
                self.i+=1
                
                
            
            
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