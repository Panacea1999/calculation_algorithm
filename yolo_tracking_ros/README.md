> This is an ROS package for detection and tracking object. Now, only person is interested

# Configuration
+ down git repository from: https://github.com/bdairobot/yolo_tracking_ros
  ```
  git clone https://github.com/bdairobot/yolo_tracking_ros vision_tracking
  ```
+ download yolo3 and tiny yolo3 from: https://pjreddie.com/darknet/yolo/
  > YOLOv3-tiny and YOLOv320
+ install packages NumPy sklean OpenCV
+ download yolo.h5 from: https://drive.google.com/file/d/1uvXFacPnrSMw6ldWTyLLjGLETlEsUvcE/view?usp=sharing
+ install Keras
+ install CUDA and KUDNN
+ install python package tensorflow-gpu using GPU or tensorflow using for CPU
+ put it in catkin workspace and then
```
catkin_make
source your setup.bash
roslaunch vision_tracking tracking.launch
```