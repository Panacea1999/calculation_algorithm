#! /usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import rospy
## CAUTION: *.py 's name should be different with package 's name
from vision_tracking.msg import Detected,DetectedArray,DetectedFull
from sensor_msgs.msg import Image as ROSImage
from sensor_msgs.msg import CompressedImage as ROSImage_C

import os
from timeit import time
import warnings
import sys
import threading
# print(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)),'src'))

## set running path to a fixed path
run_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'src')
sys.path.append(run_path)
os.chdir(run_path)

import cv2
from cv_bridge import CvBridge
import numpy as np
from PIL import Image
from yolo import YOLO

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from deep_sort.detection import Detection as ddet
warnings.filterwarnings('ignore')

class ObjectTracking():
    def __init__(self, name, metric, yolo_interface, model_filename, \
                 nms_max_overlap, image_sub, image_pub, detected_pub):
        # Set the shutdown function (stop the robot)
        # rospy.on_shutdown(self.shutdown)
        self.name = name
        self._cv_bridge = CvBridge()
        self._max_overlap = nms_max_overlap
        # deep_sort 
        self._encoder = gdet.create_box_encoder(model_filename,batch_size=1)
        self._tracker = Tracker(metric)
        self._yolo_interface = yolo_interface
        self._fps = 0.0
        
        self._image_sub = rospy.Subscriber(image_sub, ROSImage, self.image_callback, queue_size=1)
        # self.thermal_sub = rospy.Subscriber(THERMAL_TOPIC, ROSImage, self.thermal_callback, queue_size=1)

        self._image_pub = rospy.Publisher(image_pub, ROSImage, queue_size=1)
        self._detected_pub = rospy.Publisher(detected_pub, DetectedFull , queue_size=1)

        # self._cv_image = None
        self._image_msg = None
        self._image_updated = False

        self.thread = threading.Thread(target=self.process, name=name)
        # self.lock = lock

        rospy.loginfo('##################### '+name+' Initialization Finished! #####################')

    def image_callback(self, image_msg):
        # self._cv_image = self._cv_bridge.imgmsg_to_cv2(image_msg, "bgr8")
        self._image_msg = image_msg
        self._image_updated = True
        # rospy.loginfo('call back in '+self.name)

    def process(self):
        # while not rospy.is_shutdown():
        while not rospy.is_shutdown():
            now_time = time.time()

            if not self._image_updated:
                rospy.loginfo('No image in ' + self.name + ' yet! Please check the existence of Subscrided topic.')
                rospy.sleep(2.13)
                continue
            
            self._image_updated = False
            image_msg = self._image_msg

            cv_image = self._cv_bridge.imgmsg_to_cv2(image_msg, "bgr8")
            image = Image.fromarray(cv_image)
            
            # self.lock.acquire()
            boxs = self._yolo_interface.detect_image(image)
            # self.lock.release()

            # print("box_num",len(boxs))
            features = self._encoder(cv_image,boxs)
            
            # score to 1.0 here
            detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)]
            
            # Run non-maxima suppression.
            boxes = np.array([d.tlwh for d in detections])
            scores = np.array([d.confidence for d in detections])
            indices = preprocessing.non_max_suppression(boxes, self._max_overlap, scores)
            detections = [detections[i] for i in indices]
            
            # Call the _tracker
            self._tracker.predict()
            self._tracker.update(detections)
            
            detected_array = DetectedArray()
            detected_full = DetectedFull()
            detected_full.image = image_msg
            
            detected_array.size = 0
            for track in self._tracker.tracks:
                if track.is_confirmed() and track.time_since_update >1 :
                    continue 
                bbox = track.to_tlbr()
                
                detected = Detected()
                detected.object_class = 'Person'
                detected.num = np.uint32(track.track_id)
                detected.p = 1.0

                def range_check(x,min,max):
                    if x < min: x = min 
                    if x > max: x = max
                    return x
                
                #rospy.loginfo(cv_image.shape[1],cv_image.shape[0])
                print(cv_image.shape[1],cv_image.shape[0])

                detected.x = np.uint16(range_check(int(bbox[0]),0,cv_image.shape[1] - 1))
                detected.y = np.uint16(range_check(int(bbox[1]),0,cv_image.shape[0] - 1))
                detected.width = np.uint16(range_check(int(bbox[2]), 0, cv_image.shape[1] - 1)) - detected.x
                detected.height = np.uint16(range_check(int(bbox[3]), 0, cv_image.shape[0] -1 )) - detected.y

                cv2.rectangle(cv_image, (int(detected.x), int(detected.y)), (int(detected.x+detected.width), int(detected.y+detected.height)),(255,0,0), 2)
                cv2.putText(cv_image, 'Person:'+str(track.track_id),(int(detected.x), int(detected.y)),0, 5e-3 * 100, (0,255,0),2)

                detected_array.size = detected_array.size+1
                detected_array.data.append(detected)

            # for det in detections:
            #     bbox = det.to_tlbr()
            #     cv2.rectangle(cv_image,(int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,255,0), 2)
            

            # cv2.imshow('', cv_image)
            detected_full.header.stamp = image_msg.header.stamp
            detected_full.detections = detected_array
            if self._detected_pub.get_num_connections() > 0:
                self._detected_pub.publish(detected_full)

            ros_image = self._cv_bridge.cv2_to_imgmsg(cv_image)
            ros_image.header.stamp = image_msg.header.stamp
            
            if self._image_pub.get_num_connections() > 0:
                self._image_pub.publish(ros_image)
            
            self._fps  = ( self._fps + (1./(time.time()-now_time)) ) / 2
            rospy.loginfo(self.name + " processing fps = %f"%(self._fps))
        
        rospy.loginfo("Existing " + self.name + " object tracking...")

    # def shutdown(self):
    #     rospy.loginfo("Stopping the tensorflow object detection...")
        # rospy.sleep(1) 

if __name__ == '__main__':

    rospy.init_node("vision_tracking")

    rgb_topic = rospy.get_param('~rgb_topic', '/camera/left/image_raw')
    thermal_topic = rospy.get_param('~thermal_topic', '/optris/image_raw')
    rgb_tracking_flag = rospy.get_param('~rgb_tracking_flag', True)
    thermal_tracking_flag = rospy.get_param('~thermal_tracking_flag', True)

    rgb_topic_pub = rospy.get_param('~rgb_topic_pub', '~rgb_image')
    thermal_topic_pub = rospy.get_param('~thermal_topic_pub', '~thermal_image')
    rgb_detected_topic_pub = rospy.get_param('~rgb_detected_topic_pub', '~rgb_detected_full')
    thermal_detected_topic_pub = rospy.get_param('~thermal_detected_topic_pub', '~thermal_detected_full')
    # Definition of the parameters
    max_cosine_distance = 0.3
    nn_budget = None
    rgb_metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    thermal_metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)

    model_filename = 'model_data/mars-small128.pb'
    nms_max_overlap = 1.0

    # lock = threading.Lock()
    yolo_interface = YOLO()

    try:
        if rgb_tracking_flag:
            rgb_tracking = ObjectTracking('rgb', rgb_metric, yolo_interface, model_filename, \
                nms_max_overlap, rgb_topic, rgb_topic_pub, rgb_detected_topic_pub)
            rgb_tracking.thread.start()

        if thermal_tracking_flag:
            thermal_tracking = ObjectTracking('thermal', thermal_metric, yolo_interface, model_filename, \
                nms_max_overlap, thermal_topic, thermal_topic_pub, thermal_detected_topic_pub)
            # rgb_tracking.process()
            thermal_tracking.thread.start()
        if rgb_tracking_flag or thermal_tracking_flag:
            rospy.spin()
        else:
            rospy.loginfo("Nothing to be run, please check tracking enable flag param")

    except rospy.ROSInterruptException:
        rospy.loginfo("Object Tracking has started.")
