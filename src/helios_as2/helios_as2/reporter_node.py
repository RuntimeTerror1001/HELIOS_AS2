#!/usr/bin/env python3
"""
Survivor Reporter Node for ROS2 Humble
Consumes YOLO detections and produces geo-tagged survivor reports with ground plane projection.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rclpy.duration import Duration
from sensor_msgs.msg import Image, CameraInfo
from vision_msgs.msg import Detection2DArray
from geometry_msgs.msg import PointStamped
from geographic_msgs.msg import GeoPoint
from helios_interfaces.msg import SurvivorReport
from cv_bridge import CvBridge
from tf2_ros import Buffer, TransformListener
import cv2
import numpy as np
import os
import datetime
import math
import message_filters

class ReporterNode(Node):
    def __init__(self):
        super().__init__('reporter_node')

        # Set use_sim_time parameter
        self.set_parameters([rclpy.Parameter('use_sim_time', rclpy.Parameter.Type.BOOL, True)])

        # Declare parameters
        self.declare_parameter('world_frame', 'helios/map')
        self.declare_parameter('camera_frame', 'helios/rgbd_camera/rgbd_camera/camera/optical_frame')
        self.declare_parameter('ground_z', 0.0)
        self.declare_parameter('use_gps_origin', False)
        self.declare_parameter('origin_lat', 0.0)
        self.declare_parameter('origin_lon', 0.0)
        self.declare_parameter('origin_alt', 0.0)
        self.declare_parameter('save_dir', os.path.expanduser('~/.helios_reports'))
        self.declare_parameter('min_conf', 0.4)

        # Get parameters
        self.world_frame = self.get_parameter('world_frame').get_parameter_value().string_value
        self.camera_frame = self.get_parameter('camera_frame').get_parameter_value().string_value
        self.ground_z = self.get_parameter('ground_z').get_parameter_value().double_value
        self.use_gps_origin = self.get_parameter('use_gps_origin').get_parameter_value().bool_value
        self.origin_lat = self.get_parameter('origin_lat').get_parameter_value().double_value
        self.origin_lon = self.get_parameter('origin_lon').get_parameter_value().double_value
        self.origin_alt = self.get_parameter('origin_alt').get_parameter_value().double_value
        self.save_dir = self.get_parameter('save_dir').get_parameter_value().string_value
        self.min_conf = self.get_parameter('min_conf').get_parameter_value().double_value
        
        # Create save directory
        os.makedirs(self.save_dir, exist_ok=True)

        # TF Buffer and Listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # CV Bridge
        self.bridge = CvBridge()

        # Camera info storage
        self.camera_info = None
        self.camera_matrix = None

        # Detection counter for unique IDs
        self.detection_counter = 0

        # QoS Profile
        reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            depth=10
        )

        # Subscribers
        self.detection_sub = self.create_subscription(
            Detection2DArray,
            '/helios/perception/result',
            self.detection_callback,
            10
        )

        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            '/helios/camera/color/camera_info',
            self.camera_info_callback,
            reliable_qos
        )

        # Synchronized subscribers for depth and color images
        self.color_sub = message_filters.Subscriber(
            self, Image, '/helios/camera/color/image_raw',
            qos_profile=reliable_qos
        )

        self.depth_sub = message_filters.Subscriber(
            self, Image, '/helios/camera/depth/image_raw',
            qos_profile=reliable_qos
        )

        # Approximate time synchronizer
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [self.color_sub, self.depth_sub],
            queue_size=50,
            slop=0.05
        )
        self.sync.registerCallback(self.synchronized_callback)

        # Publisher
        self.report_pub = self.create_publisher(
            SurvivorReport,
            '/helios/people/report',
            10
        )

        # Storage for latest images
        self.latest_color_image = None
        self.latest_depth_image = None
        self.latest_image_stamp = None
        
        self.get_logger().info(f"Reporter node initialized. Save directory: {self.save_dir}")

    def camera_info_callback(self, msg):
        """
        Store camera calibration info
        """
        if self.camera_info is None:
            self.camera_info = msg
            self.camera_matrix = np.array(msg.k).reshape(3, 3)
            self.get_logger().info("Camera calibration received")
    
    def synchronized_callback(self, color_msg, depth_msg):
        """
        Store synchronized color and depth images
        """
        try:
            self.latest_color_image = self.bridge.imgmsg_to_cv2(color_msg, 'bgr8')
            self.latest_depth_image = self.bridge.imgmsg_to_cv2(depth_msg, 'passthrough')
            self.latest_image_stamp = color_msg.header.stamp
        except Exception as e:
            self.get_logger().warn(f"Error converting images: {e}")

    def detection_callback(self, msg):
        """
        Process detections and create survivor reports
        """
        if not msg.detections:
            return
        
        if self.camera_info is None:
            self.get_logger().warn('No camera info available, skipping detections')
            return
        
        for detection in msg.detections:
            # Check confidence threshold
            if detection.results and len(detection.results) > 0:
                conf = detection.results[0].hypothesis.score
                if conf < self.min_conf:
                    continue
            else:
                continue
            
            try:
                # Create survivor report
                report = self.create_survivor_report(detection, msg.header)
                if report:
                    self.report_pub.publish(report)
                    self.get_logger().info(f"Published survivor report: {report.det_id}")
                    
            except Exception as e:
                self.get_logger().error(f"Error processing detection: {e}")

    def create_survivor_report(self, detection, header):
        """Create a SurvivorReport from a detection"""
        try:
            # Get detection center
            cx = int(detection.bbox.center.position.x)
            cy = int(detection.bbox.center.position.y)
            
            # Get confidence
            conf = detection.results[0].hypothesis.score if detection.results else 0.0
            
            # Create unique detection ID
            self.detection_counter += 1
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            det_id = f"survivor_{timestamp}_{self.detection_counter:04d}"
            
            # Project to 3D world coordinates
            world_point = self.project_to_world(cx, cy, header.stamp)
            if world_point is None:
                self.get_logger().warn("Failed to project detection to world frame")
                return None
            
            # Create SurvivorReport
            report = SurvivorReport()
            report.header = header
            report.header.frame_id = self.world_frame
            
            # Set map pose
            report.map_pose.header = report.header
            report.map_pose.pose.position.x = world_point.x
            report.map_pose.pose.position.y = world_point.y
            report.map_pose.pose.position.z = world_point.z
            # Identity quaternion for orientation
            report.map_pose.pose.orientation.w = 1.0
            
            # Set geopoint
            if self.use_gps_origin:
                report.geopoint = self.enu_to_wgs84(world_point.x, world_point.y, world_point.z)
            else:
                report.geopoint.latitude = 0.0
                report.geopoint.longitude = 0.0
                report.geopoint.altitude = 0.0
            
            # Set other fields
            report.confidence = conf
            report.det_id = det_id
            report.detection_time = self.get_clock().now().to_msg()
            
            # Save annotated image
            report.image_path = self.save_annotated_image(detection, det_id)
            
            return report
            
        except Exception as e:
            self.get_logger().error(f"Error creating survivor report: {e}")
            return None
    
    def project_to_world(self, pixel_x, pixel_y, stamp):
        """Project pixel coordinates to world frame"""
        try:
            # Create point in camera optical frame
            camera_point = PointStamped()
            camera_point.header.stamp = stamp
            camera_point.header.frame_id = self.camera_frame
            
            # Try to get depth if available
            if (self.latest_depth_image is not None and 
                self.latest_image_stamp is not None):
                
                # Check if depth image is recent enough
                stamp_sec = stamp.sec + stamp.nanosec * 1e-9
                latest_sec = self.latest_image_stamp.sec + self.latest_image_stamp.nanosec * 1e-9
                
                if abs(stamp_sec - latest_sec) < 0.1:  # 100ms tolerance
                    # Get depth value
                    depth = self.get_depth_at_pixel(pixel_x, pixel_y)
                    if depth > 0:
                        # Back-project using depth
                        camera_point.point = self.backproject_pixel(pixel_x, pixel_y, depth)
                    else:
                        # Ray casting to ground plane
                        camera_point.point = self.raycast_to_ground(pixel_x, pixel_y, stamp)
                else:
                    # Ray casting to ground plane
                    camera_point.point = self.raycast_to_ground(pixel_x, pixel_y, stamp)
            else:
                # Ray casting to ground plane
                camera_point.point = self.raycast_to_ground(pixel_x, pixel_y, stamp)
            
            # Transform to world frame
            world_point = self.tf_buffer.transform(camera_point, self.world_frame, timeout=Duration(seconds=1.0))
            return world_point.point
            
        except Exception as e:
            self.get_logger().error(f"Error projecting to world: {e}")
            return None
    
    def get_depth_at_pixel(self, x, y):
        """Get depth value at pixel coordinates"""
        if self.latest_depth_image is None:
            return 0.0
        
        h, w = self.latest_depth_image.shape[:2]
        if 0 <= x < w and 0 <= y < h:
            depth = self.latest_depth_image[y, x]
            # Convert mm to m if needed
            if depth > 100:  # Assume mm if > 100
                depth = depth / 1000.0
            return float(depth)
        return 0.0
    
    def backproject_pixel(self, x, y, depth):
        """Back-project pixel to 3D point using depth"""
        if self.camera_matrix is None:
            raise ValueError("Camera matrix not available")
        
        # Intrinsic parameters
        fx = self.camera_matrix[0, 0]
        fy = self.camera_matrix[1, 1]
        cx = self.camera_matrix[0, 2]
        cy = self.camera_matrix[1, 2]
        
        # Back-project to camera coordinates
        from geometry_msgs.msg import Point
        point = Point()
        point.z = float(depth)
        point.x = (x - cx) * depth / fx
        point.y = (y - cy) * depth / fy
        
        return point
    
    def raycast_to_ground(self, x, y, stamp):
        """Cast ray from pixel to ground plane intersection"""
        if self.camera_matrix is None:
            raise ValueError("Camera matrix not available")
        
        # Get camera pose in world frame
        try:
            transform = self.tf_buffer.lookup_transform(
                self.world_frame, 
                self.camera_frame,
                stamp,
                timeout=rclpy.duration.Duration(seconds=1.0)
            )
        except Exception as e:
            self.get_logger().warn(f"Failed to get camera transform: {e}")
            # Return a default point
            from geometry_msgs.msg import Point
            point = Point()
            point.x = 0.0
            point.y = 0.0
            point.z = self.ground_z
            return point
        
        # Camera position in world frame
        cam_x = transform.transform.translation.x
        cam_y = transform.transform.translation.y
        cam_z = transform.transform.translation.z
        
        # Ray direction in camera frame (normalized)
        fx = self.camera_matrix[0, 0]
        fy = self.camera_matrix[1, 1]
        cx = self.camera_matrix[0, 2]
        cy = self.camera_matrix[1, 2]
        
        ray_x = (x - cx) / fx
        ray_y = (y - cy) / fy
        ray_z = 1.0
        
        # Normalize ray direction
        ray_norm = math.sqrt(ray_x*ray_x + ray_y*ray_y + ray_z*ray_z)
        ray_x /= ray_norm
        ray_y /= ray_norm
        ray_z /= ray_norm
        
        # Transform ray direction to world frame
        # (simplified - assumes camera is level for now)
        world_ray_x = ray_x
        world_ray_y = ray_y
        world_ray_z = ray_z
        
        # Intersect with ground plane (z = ground_z)
        if abs(world_ray_z) < 1e-6:
            # Ray is parallel to ground, use camera position
            intersect_x = cam_x
            intersect_y = cam_y
        else:
            t = (self.ground_z - cam_z) / world_ray_z
            intersect_x = cam_x + t * world_ray_x
            intersect_y = cam_y + t * world_ray_y
        
        from geometry_msgs.msg import Point
        point = Point()
        point.x = intersect_x
        point.y = intersect_y
        point.z = self.ground_z
        
        return point
    
    def enu_to_wgs84(self, x, y, z):
        """Convert ENU coordinates to WGS84 (simplified)"""
        # This is a simplified conversion - for production use proper geodetic libraries
        geopoint = GeoPoint()
        
        # Approximate conversion (assumes small distances)
        lat_per_m = 1.0 / 111320.0  # degrees per meter latitude
        lon_per_m = 1.0 / (111320.0 * math.cos(math.radians(self.origin_lat)))  # degrees per meter longitude
        
        geopoint.latitude = self.origin_lat + y * lat_per_m
        geopoint.longitude = self.origin_lon + x * lon_per_m
        geopoint.altitude = self.origin_alt + z
        
        return geopoint
    
    def save_annotated_image(self, detection, det_id):
        """Save annotated image with detection highlighted"""
        if self.latest_color_image is None:
            return ""
        
        try:
            # Create annotated image
            annotated = self.latest_color_image.copy()
            
            # Draw bounding box
            cx = int(detection.bbox.center.position.x)
            cy = int(detection.bbox.center.position.y)
            w = int(detection.bbox.size_x)
            h = int(detection.bbox.size_y)
            
            x1 = cx - w//2
            y1 = cy - h//2
            x2 = cx + w//2
            y2 = cy + h//2
            
            # Draw detection
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 3)
            
            # Add label
            conf = detection.results[0].hypothesis.score if detection.results else 0.0
            label = f"Survivor {conf:.2f}"
            cv2.putText(annotated, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Save image
            filename = f"{det_id}.jpg"
            filepath = os.path.join(self.save_dir, filename)
            cv2.imwrite(filepath, annotated)
            
            return filepath
            
        except Exception as e:
            self.get_logger().error(f"Error saving annotated image: {e}")
            return ""


def main(args=None):
    rclpy.init(args=args)
    node = ReporterNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()