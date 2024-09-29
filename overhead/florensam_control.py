# tennis ball should use florence2 base
# this script tracks the ball, move the ball to person or to a fixed position, and release the ball.
# the process is done repeatedly until killed.

import os
from dataclasses import dataclass
from typing import Any
import sys

import cv2
import numpy as np
import supervision as sv
import torch
from autodistill.detection import CaptionOntology, DetectionBaseModel
from autodistill.helpers import load_image
from florence_model import Florence2

import time
import socket

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DOG_IP = "127.0.0.1"
CAM_ID = 0
APPROACH = 0
PICK_UP = 1
DROP_OFF = -1
RELEASE = 512
TARGET_POINT = [700,450]
TRACK_HUMAN = True
import tty
import termios

def getch():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

checkpoint = "../segment-anything-2/checkpoints/sam2_hiera_base_plus.pth"
model_cfg = "sam2_hiera_b+.yaml"
SamPredictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))


@dataclass
class FlorenceSAM2(DetectionBaseModel):
    ontology: CaptionOntology
    florence_2_predictor: Florence2
    box_threshold: float
    text_threshold: float

    def __init__(self, ontology: CaptionOntology):
        self.ontology = ontology
        self.florence_2_predictor = Florence2(ontology=ontology)
        self.sam_2_predictor = SamPredictor
        self.time_log = []
        self.init_frame = True

    def predict(self, input) -> sv.Detections:
        predict_start_time = time.time()
        if self.init_frame:
            self.florence_2_detections = self.florence_2_predictor.predict(input)
            self.init_frame = False

        florence_time = time.time()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            self.sam_2_predictor.set_image(input)
            result_masks = []
            for box in self.florence_2_detections.xyxy:
                masks, scores, _ = self.sam_2_predictor.predict(
                    box=box, multimask_output=False
                )
                index = np.argmax(scores)
                masks = masks.astype(bool)
                result_masks.append(masks[index])

            result_masks_np = np.array(result_masks)
            self.florence_2_detections.mask = result_masks_np
            self.florence_2_detections.xyxy = sv.mask_to_xyxy(result_masks_np)  
        sam2_time = time.time()
        self.time_log.append([florence_time-predict_start_time, sam2_time-florence_time])
        return self.florence_2_detections
    
class OverHeadController:
    def __init__(self):
        self.base_model = FlorenceSAM2(ontology=CaptionOntology({  # label:prompt
                    "tennis ball": "tennis ball",
                    "person": "person",
                    "robot" : "robot",
                }))
        self.frame_id = 0
        self.detect_interval = 20
        with np.load('calib_result.npz') as X:
            self.DIM, self.K, self.D = [X[i] for i in ('DIM','K', 'D')]
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.current_state = APPROACH
        self.map1, self.map2 = cv2.fisheye.initUndistortRectifyMap(self.K, self.D, np.eye(3), self.K, self.DIM, cv2.CV_16SC2)

    def undistort(self,img):
        undistorted_img = cv2.remap(img, self.map1, self.map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)    
        return undistorted_img
    
    def compute_pose(self, result, frame):
        object_mask = result.mask[0]
        person_mask = result.mask[1]
        robot_mask = result.mask[2]
        H, W = robot_mask.shape
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_red1 = np.array([0, 0, 0])     # Lower bound for red hue
        upper_red1 = np.array([20, 255, 255])  # Upper bound for red hue
        lower_red2 = np.array([140, 0, 0])   # Second lower bound for red hue
        upper_red2 = np.array([180, 255, 255]) # Second upper bound for red hue
        red_mask1 = cv2.inRange(hsv_frame, lower_red1, upper_red1)
        red_mask2 = cv2.inRange(hsv_frame, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        
        object_pos, person_pos, robot_pos, head_pos, robot_box, head_box = None, None, None, None, None, None
        
        if np.max(object_mask) == 0:
            object_pos = None
        else:
            img = (object_mask > 0).astype(np.uint8)*255
            moments = cv2.moments(img)
            px = int(moments['m10']/moments['m00'])
            py = int(moments['m01']/moments['m00'])
            object_pos = (px,py)

        if np.max(person_mask) == 0:
            person_pos = None
        else:
            img = (person_mask > 0).astype(np.uint8)*255
            moments = cv2.moments(img)
            px = int(moments['m10']/moments['m00'])
            py = int(moments['m01']/moments['m00'])
            person_pos = (px,py) 
        if np.max(robot_mask) == 0:
            robot_pos = None
            head_pos = None
            robot_box = None
            head_box = None
        else:
            img = (robot_mask > 0).astype(np.uint8)*255
            moments = cv2.moments(img)
            px = int(moments['m10']/moments['m00'])
            py = int(moments['m01']/moments['m00'])
            robot_pos = (px,py)

            thresh = ((robot_mask>0)*255).astype(np.uint8)
            # find the tilted bounding box
            contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, 
                                        cv2.CHAIN_APPROX_SIMPLE) 
            cnt_shape = np.array([c.shape[0] for c in contours])
            cnt_id = np.argmax(cnt_shape)
            cnt = contours[cnt_id] 
            rect = cv2.minAreaRect(cnt) 
            box = cv2.boxPoints(rect)
            box = np.intp(box)
            side_a = np.linalg.norm(box[1] - box[0])
            side_b = np.linalg.norm(box[1] - box[2])  
            
            # color based method to determine the head position
            if side_a > side_b:
                detect_box_a = np.array([box[1], box[2], 0.5*(box[2]+box[3]), 0.5*(box[0]+box[1])])
                detect_box_b = np.array([box[0], box[3], 0.5*(box[2]+box[3]), 0.5*(box[0]+box[1])])

                head_mask_a = cv2.fillPoly(np.zeros((H,W), np.uint8), [detect_box_a.astype(np.int32)], 255)
                head_mask_b = cv2.fillPoly(np.zeros((H,W), np.uint8), [detect_box_b.astype(np.int32)], 255)
                # compute the area of red pixels in the mask and frame
                head_area_a = np.sum(thresh*head_mask_a*red_mask)
                head_area_b = np.sum(thresh*head_mask_b*red_mask)
                if head_area_a > head_area_b:
                    head_box = np.intp(detect_box_a)
                    head_pos = np.sum(detect_box_a, axis=0) / 4
                else:
                    head_box = np.intp(detect_box_b)
                    head_pos = np.sum(detect_box_b, axis=0) / 4

            else:
                detect_box_a = np.array([box[1], box[0], 0.5*(box[0]+box[3]), 0.5*(box[2]+box[1])])
                detect_box_b = np.array([box[2], box[3], 0.5*(box[0]+box[3]), 0.5*(box[2]+box[1])])

                head_mask_a = cv2.fillPoly(np.zeros((H,W), np.uint8), [detect_box_a.astype(np.int32)], 255)
                head_mask_b = cv2.fillPoly(np.zeros((H,W), np.uint8), [detect_box_b.astype(np.int32)], 255)
                # compute the area of red pixels in the mask and frame
                head_area_a = np.sum(thresh*head_mask_a*red_mask)
                head_area_b = np.sum(thresh*head_mask_b*red_mask)
                if head_area_a > head_area_b:
                    head_box = np.intp(detect_box_a)
                    head_pos = np.sum(detect_box_a, axis=0) / 4
                else:
                    head_box = np.intp(detect_box_b)
                    head_pos = np.sum(detect_box_b, axis=0) / 4

            robot_box = box
        
        return object_pos, person_pos, robot_pos, (int(head_pos[0]), int(head_pos[1])), robot_box

    def loop(self):
        print("start loop")

        # load video
        video_path = 0# 'sample_video.mp4'
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        if not cap.isOpened():
            print("Error: Could not open video.")
            exit()

        start_time = time.time()
        result_imgs = []
        cmd_hist = []
        dist_hist = []
        prev_head_rel_vec = None
        num_frames = -1

        while True:
            loop_start_time = time.time()
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame.")
                break

            # preprocess image
            frame = self.undistort(frame)
            H, W, _ = frame.shape
            cv2.imshow('live',frame)

            # detect and predict
            if self.frame_id % self.detect_interval == 0:
                self.base_model.init_frame = True
            results = self.base_model.predict(frame)
            
            # compute pose
            if self.current_state == APPROACH:
                if len(results.mask) < 2:  # TODO: more complex rules
                    print(f"Not enough objects detected. {self.frame_id=}. {len(results.mask)=}")
                    self.base_model.init_frame = True
                    continue
                bpos, ppos, rpos, headpos, rbox = self.compute_pose(results,frame)
                if bpos is not None:
                    cv2.circle(frame, bpos, 2, (255,0,0), -1)
                if rpos is not None:
                    cv2.circle(frame, rpos, 5, (0,0,255), -1)
                    cv2.polylines(frame, [rbox], isClosed=True, color=(0, 255, 0), thickness=2) 
                    cv2.arrowedLine(frame, rpos, headpos, (0, 120, 255), 2, cv2.LINE_AA)
            elif self.current_state == DROP_OFF:
                bpos, ppos, rpos, headpos, rbox = self.compute_pose(results,frame)
                if TRACK_HUMAN:
                    bpos = ppos
                else:
                    bpos = TARGET_POINT
                if bpos is not None:
                    cv2.circle(frame, bpos, 10, (0,255,0), -1)
                if rpos is not None:
                    cv2.circle(frame, rpos, 5, (0,0,255), -1)
                    cv2.polylines(frame, [rbox], isClosed=True, color=(0, 255, 0), thickness=2) 
                    cv2.arrowedLine(frame, rpos, headpos, (0, 120, 255), 2, cv2.LINE_AA)

            # compute command
            if bpos is not None and rpos is not None:
                ball_rel_vec = np.array(bpos) - np.array(rpos)
                head_rel_vec = np.array(headpos) - np.array(rpos)
                if prev_head_rel_vec is not None:
                    if np.dot(head_rel_vec, prev_head_rel_vec) < 0:
                        self.base_model.init_frame = True  # TODO: more complex rules
                        print("wrong direction")
                        continue
                prev_head_rel_vec = head_rel_vec.copy()

                dist = np.linalg.norm(ball_rel_vec)
                diff_x = np.dot(ball_rel_vec, head_rel_vec)/np.linalg.norm(head_rel_vec) #- np.linalg.norm(head_rel_vec)
                # diff_y = ball_rel_vec - diff_x*head_rel_vec/np.linalg.norm(head_rel_vec)
                diff_y = np.cross(ball_rel_vec, head_rel_vec)/np.linalg.norm(head_rel_vec)
                diff_w = np.arctan2(diff_y, diff_x)
                print(f"{diff_w=}")
                if diff_x > 15 and abs(diff_w) <= 0.4:  #30
                    cmd_x = 0.6
                elif diff_x > 15 and abs(diff_w) > 0.4:
                    cmd_x = 0.5
                elif diff_x < -15:
                    cmd_x = -0.5
                else:
                    cmd_x = 0
                if abs(diff_w) > 0.6:
                    cmd_x = -0.3

                if diff_y > 15:
                    cmd_y = -0.3*0
                elif diff_y < -15:
                    cmd_y = 0.3*0
                else:
                    cmd_y = 0
                cmd_r = np.clip(-1*diff_w, -0.7, 0.7) # negative because cmd omega is timed -1 in subscription for the sake of the joystick
                cmd_r *= np.abs(cmd_r)>0.05 # 0.2
                cmd = [cmd_x, cmd_r, 0]
            else:
                print("ball or robot not found")
                cmd = [0, 0, 0]

            # update state and send command
            if self.current_state == APPROACH:
                if dist < 230 and abs(diff_w) < 25/180*np.pi:
                    self.current_state = PICK_UP
                    cmd = [0, 0, 0]
                cmd.append(self.current_state)
                print(f"cmd:{cmd}")
                print(f"dist:{dist:.2f}")  # 230 seems good for 1280x720
                self.sock.sendto(str(cmd).encode('utf-8'), (DOG_IP, 8888))
            elif self.current_state == PICK_UP:
                # sys.exit()
                cmd = [0, 0, 0, self.current_state]
                self.sock.sendto(str(cmd).encode('utf-8'), (DOG_IP, 8888))
                print("Press enter to continue to give the object to target point")
                getch()
                self.current_state = DROP_OFF
                self.base_model.init_frame = True
            elif self.current_state == DROP_OFF:
                if TRACK_HUMAN:
                    dist_th  = 300
                    close_to_target = dist < dist_th and abs(diff_w) < 30/180*np.pi
                else:
                    dist_th  = 100
                    close_to_target = dist < dist_th
                if close_to_target:
                    self.current_state = RELEASE
                    cmd = [0, 0, -1.0]
                    release_time = time.time()
                cmd.append(0)
                print(f"cmd:{cmd}")
                print(f"dist:{dist:.2f}")  # 230 seems good for 1280x720
                self.sock.sendto(str(cmd).encode('utf-8'), (DOG_IP, 8888))
            elif self.current_state == RELEASE:
                if time.time() - release_time > 2:
                    cmd = [0, 0, -1.0, RELEASE]
                    self.sock.sendto(str(cmd).encode('utf-8'), (DOG_IP, 8888))
                    print("end of one trial")
                if time.time() - release_time > 3:
                    cmd = [0, 0, 0, RELEASE]
                    self.sock.sendto(str(cmd).encode('utf-8'), (DOG_IP, 8888))
                    if not hasattr(self, 'start_next_trial') or not self.start_next_trial:
                        print("Press enter to continue to the next trial")
                        getch()
                        self.start_next_trial = True
                        self.start_next_trial_time = time.time()
                        self.base_model.init_frame = True
                if hasattr(self, 'start_next_trial') and self.start_next_trial:
                    if time.time() - self.start_next_trial_time > 2:
                        self.start_next_trial = False
                        self.current_state = APPROACH


            # annotate images for visualization
            mask_annotator = sv.MaskAnnotator()
            annotated_image = mask_annotator.annotate(frame.copy(), detections=results)
            if len(result_imgs)<600:  # save 
                result_imgs.append(annotated_image)
                cmd_hist.append(cmd)
                dist_hist.append(dist)
            cv2.imshow('annotated',annotated_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            self.frame_id += 1
            if self.frame_id == num_frames:
                break

            time.sleep(max(0,1/20 - (time.time() - loop_start_time)))

        end_time = time.time()
        print(f"total frames: {self.frame_id}")
        print(f"total time: {end_time-start_time:.2f}s")
        print("fps:", self.frame_id/(end_time-start_time))
        time_log = np.array(self.base_model.time_log)
        print("avg florence time:", np.mean(time_log[:,0]))
        print("avg sam2 time:", np.mean(time_log[:,1]))

        for i, img in enumerate(result_imgs):
            cv2.imwrite(f"../florence-sam2/results/res_{i}_{cmd_hist[i][0]}_{cmd_hist[i][2]:.2f}_dist{dist_hist[i]:.2f}.jpg", img)

        print("saving video...")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can use other codecs like 'MJPG', 'MP4V', etc.
        fps = 20.0
        undis_frame_size = (result_imgs[0].shape[1], result_imgs[0].shape[0])  # Width and height of the frame
        undis_out = cv2.VideoWriter("../florence-sam2/results/result_output.mp4", fourcc, fps, undis_frame_size)
        for frame in result_imgs:
            undis_out.write(frame)
        undis_out.release()

if __name__ == '__main__':
    ctlr = OverHeadController()
    ctlr.loop()