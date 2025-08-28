#!/usr/bin/env python3
"""
Mission Patrol using Aerostack2 Python API (DroneInterface)
- Arms, switches to offboard, takes off
- Flies a square patrol at given height & size
- Lands on completion (or Ctrl-C)

Requires: `as2_python_api` to be installed (part of Aerostack2)
"""
import argparse
import sys
import time
from typing import List

import rclpy
from rclpy.node import Node
from rclpy.utilities import remove_ros_args
from geometry_msgs.msg import Point

# Aerostack2 Python API
from as2_python_api.drone_interface import DroneInterface


class MissionPatrol(Node):
    def __init__(self, args):
        super().__init__('helios_patrol_mission')
        self.args = args
        self.get_logger().info('Initializing DroneInterface...')
        self.drone = DroneInterface(drone_id=self.args.namespace)

    # ---------- helpers ----------
    def wait_for_platform_ready(self, timeout=30.0) -> bool:
        self.get_logger().info('Waiting for platform status...')
        deadline = time.time() + timeout
        while rclpy.ok() and time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            try:
                info = self.drone.info # dict : connected, armed, offboard, state, etc.
            except Exception as e:
                self.get_logger().warn(f'Cannot read platform info yet: {e}')
                info = None
            if info and info.get('connected', False):
                self.get_logger().info(f'Platform connected: {info}')
                return True
            time.sleep(0.2)
        
        return False


    def takeoff(self) -> bool:
        self.get_logger().info(f'Takeoff to {self.args.height:.2f} m (speed {self.args.speed:.2f} m/s)')
        try:
            self.drone.arm()
            time.sleep(1.0)
            self.drone.offboard()
            ok = self.drone.takeoff(height=self.args.height, speed=self.args.speed, wait=True)
            if not ok:
                self.get_logger().error('Takeoff returned False')
            return ok
        except Exception as e:
            self.get_logger().error(f'Takeoff error: {e}')
            return False

    def land(self) -> bool:
        self.get_logger().info('Landing...')
        try:
            ok = self.drone.land(wait=True)
            return ok
        except Exception as e:
            self.get_logger().error(f'Land error: {e}')
            return False

    def goto_point(self, x: float, y: float, z: float, timeout: float = 40.0) -> bool:
        try:
            pt = Point(x=x, y=y, z=z)
            self.get_logger().info(f'GoTo → ({x:.1f}, {y:.1f}, {z:.1f}) in frame "earth"')
            # Aerostack2 frame: use 'earth'
            ok = self.drone.go_to.go_to_point(point=pt, frame_id='earth', yaw=None, speed=self.args.speed, wait=True)
            if not ok:
                self.get_logger().warn('GoTo returned False')
            return ok
        except Exception as e:
            self.get_logger().error(f'GoTo error: {e}')
            return False

    # ---------- patterns ----------
    def square_waypoints(self) -> List[Point]:
        s = self.args.size
        z = self.args.height
        # square centered at origin (earth frame). Adjust as needed.
        return [
            Point(x= 0.0, y= 0.0, z=z),
            Point(x= s,   y= 0.0, z=z),
            Point(x= s,   y= s,   z=z),
            Point(x= 0.0, y= s,   z=z),
            Point(x= 0.0, y= 0.0, z=z)
        ]

    # ---------- mission ----------
    def run(self) -> int:
        if not self.wait_for_platform_ready(30.0):
            self.get_logger().error('Platform not ready in time')
            return 1
        if not self.takeoff():
            return 2

        # Patrol pattern
        if self.args.pattern == 'square':
            wps = self.square_waypoints()
        else:
            self.get_logger().warn(f'Unknown pattern "{self.args.pattern}", defaulting to square')
            wps = self.square_waypoints()

        for i, p in enumerate(wps):
            if not self.goto_point(p.x, p.y, p.z):
                self.get_logger().error(f'Failed to reach WP#{i}')
                self.land()
                return 3
            time.sleep(self.args.loiter)

        self.get_logger().info('Pattern complete.')
        self.land()
        return 0


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description='Helios SAR mission using Aerostack2')
    ap.add_argument('--namespace', '-n', default='helios')
    ap.add_argument('--height', '-H', type=float, default=3.0)
    ap.add_argument('--speed', '-s', type=float, default=1.0)
    ap.add_argument('--size', type=float, default=10.0)
    ap.add_argument('--pattern', choices=['square'], default='square')
    ap.add_argument('--loiter', type=float, default=2.0)
    ap.add_argument('--autostart', action='store_true')

    non_ros = remove_ros_args(sys.argv)
    return ap.parse_args(non_ros[1:])


def main(argv=None):
    args = parse_args(argv)
    rclpy.init(args=sys.argv)
    node = MissionPatrol(args)

    ret = 0
    try:
        if args.autostart:
            ret = node.run()
        else:
            node.get_logger().info('Mission loaded. Start with --autostart or extend to add a service trigger.')
            rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Interrupted. Landing...')
        try:
            node.land()
        except Exception:
            pass
        ret = 0
    finally:
        node.destroy_node()
        rclpy.shutdown()
    sys.exit(ret)


if __name__ == '__main__':
    main()