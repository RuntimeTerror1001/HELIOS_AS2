#!/usr/bin/env python3
"""
YOLOv8 People Detection Node for ROS2
Detects people using YOLOv8 and publishes vision_msgs/Detection2DArray with optional annotated images.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO

class PeopleDetectorNode(Node):
    def __init__(self):
        super().__init__('people_detector_node')

        # Set use_sim_time parameter
        self.set_parameters([rclpy.Parameter('use_sim_time', rclpy.Parameter.Type.BOOL, True)])

        # Declare parameters
        self.declare_parameter('model_path', '/home/redpaladin/Projects/helios_as2/src/helios_as2/detection/yolov8n.pt')
        self.declare_parameter('device', 'cuda')
        self.declare_parameter('conf', 0.4)
        self.declare_parameter('annotate', True)

        # Get parameters
        model_path = self.get_parameter('model_path').get_parameter_value().string_value
        device = self.get_parameter('device').get_parameter_value().string_value
        self.conf_thres = self.get_parameter('conf').get_parameter_value().double_value
        self.annotate = self.get_parameter('annotate').get_parameter_value().bool_value

        self.model = YOLO(model_path)
        self.model.fuse()
        self.model.to(device)
        self.get_logger().info(f'Loaded YOLOv8n model: {model_path} on device: {device}')

        self.bridge = CvBridge()

        # QoS profile for reliable image transport
        image_qos = QoSProfile(
            reliability = ReliabilityPolicy.RELIABLE,
            depth=10
        )

        # Subscribers
        self.image_sub = self.create_subscription(
            Image,
            '/helios/camera/color/image_raw',
            self.image_callback,
            image_qos
        )

        # Publisher
        self.detection_pub = self.create_publisher(
            Detection2DArray,
            '/helios/perception/result',
            10
        )

        if self.annotate:
            self.annotated_pub = self.create_publisher(
                Image,
                '/helios/perception/annotated',
                10
            )
        
        self.get_logger().info('People Detection Node initialized')

    def image_callback(self, msg):
        try:
            # Convert ROS image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

            # Run YOLO detection
            results = self.model(cv_image, conf= self.conf_thres)

            # Create detection message
            detection_array = Detection2DArray()
            detection_array.header = msg.header
            detection_array.detections = []

            # Process results
            if len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes

                for box in boxes:
                    cls = int(box.cls[0])
                    if cls == 0:
                        conf = float(box.conf[0])

                        # Extract bounding box
                        x1, y1, x2, y2 = box.xyxy[0].tolist()

                        # Create detection message
                        detection = Detection2D()
                        detection.header = msg.header

                        # Set bouding box
                        detection.bbox.center.position.x = (x1 + x2) / 2
                        detection.bbox.center.position.y = (y1 + y2) / 2
                        detection.bbox.size_x = x2 - x1
                        detection.bbox.size_y = y2 - y1

                        # Set hypothesis
                        hypothesis = ObjectHypothesisWithPose()
                        hypothesis.hypothesis.class_id = str(cls)
                        hypothesis.hypothesis.score = conf
                        detection.results = [hypothesis]

                        detection_array.detections.append(detection)
            
            # Publish detections
            self.detection_pub.publish(detection_array)

            # Create and publish annotated images if enabled
            if self.annotate and len(detection_array.detections) > 0:
                annotated_image = self.draw_detections(cv_image, detection_array.detections)
                annotated_msg = self.bridge.cv2_to_imgmsg(annotated_image, 'bgr8')
                annotated_msg.header = msg.header
                self.annotated_pub.publish(annotated_msg)
            elif self.annotate:
                # Publish original image if no detections
                annotated_msg = self.bridge.cv2_to_imgmsg(cv_image, 'bgr8')
                annotated_msg.header = msg.header
                self.annotated_pub.publish(annotated_msg)

        except Exception as e:
            self.get_logger().error(f'Error in image callback: {e}')
    
    def draw_detections(self, image, detections):
        """
        Draw bounding boxes and labels on image
        """
        annotated = image.copy()

        for detection in detections:
            # Extract bounding box
            cx = detection.bbox.center.position.x
            cy = detection.bbox.center.position.y
            w = detection.bbox.size_x
            h = detection.bbox.size_y

            x1 = int(cx - w/2)
            x2 = int(cx + w/2)
            y1 = int(cy - h/2)
            y2 = int(cy + h/2)

            # Get confidence
            conf = detection.results[0].hypothesis.score if detection.results else 0.0

            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw label
            label = f'Person: {conf:.2f}'
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(annotated, (x1, y1 - label_h - 10), (x1 + label_w, y1), (0, 255, 0), -1)
            cv2.putText(annotated, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        return annotated


def main(args=None):
    rclpy.init(args=args)
    node = PeopleDetectorNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()