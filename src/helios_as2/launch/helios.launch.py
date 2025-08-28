
#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
#
# Helios unified launch (Gazebo + Aerostack2 + app nodes)
#
# Notes:
# - Explicitly forwards a plugin_name to each AS2 include to avoid collisions
#   in the shared launch context (see aerostack2/aerostack2#733).
# - Defaults are usable out-of-the-box for Gazebo + ground_truth + differential_flatness_controller.
#
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, GroupAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node, PushRosNamespace
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # ----------------------
    # Common arguments
    # ----------------------
    namespace = LaunchConfiguration('namespace')
    params_file = LaunchConfiguration('params_file')

    declare_namespace = DeclareLaunchArgument(
        'namespace', default_value='helios', description='ROS namespace for the UAV'
    )
    declare_params = DeclareLaunchArgument(
        'params_file',
        default_value=PathJoinSubstitution([
            get_package_share_directory('helios_as2'), 'config', 'params.yaml'
        ]),
        description='Helios application parameters YAML'
    )

    # ----------------------
    # Gazebo (ros_gz_sim) world
    # ----------------------

    sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([FindPackageShare('as2_gazebo_assets'), 'launch', 'launch_simulation.py'])
        ),
        launch_arguments={
            'simulation_config_file': '/home/redpaladin/Projects/helios_as2/src/helios_as2/config/helios_sim.json', 
            'use_sim_time': 'true',
            'headless': 'false',
            'run_on_start': 'true',
        }.items()
    )

    # ----------------------
    # ros_gz_bridge (minimal)
    # ----------------------
    # Bridge /clock and image topics (extend as needed)
    # bridge_clock = Node(
    #     package='ros_gz_bridge',
    #     executable='parameter_bridge',
    #     name='bridge_clock',
    #     arguments=['/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock]'],
    #     output='screen'
    # )

    # ----------------------
    # Aerostack2 core includes
    # ----------------------
    # AS2 Platform (Gazebo)
    as2_platform_share = get_package_share_directory('as2_platform_gazebo')
    platform_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(as2_platform_share, 'launch', 'platform_gazebo_launch.py')),
        launch_arguments={
            'use_sim_time': 'true',
            'create_bridges': 'true'
        }.items()
    )

    # AS2 State Estimator
    as2_state_share = get_package_share_directory('as2_state_estimator')
    state_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(as2_state_share, 'launch', 'state_estimator_launch.py')),
        launch_arguments={
            'use_sim_time': 'true',
            'plugin_name': 'ground_truth'
        }.items()
    )

    # AS2 Motion Controller
    as2_ctrl_share = get_package_share_directory('as2_motion_controller')
    controller_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(as2_ctrl_share, 'launch', 'controller_launch.py')),
        launch_arguments={
            'use_sim_time': 'true',
            'plugin_name': 'pid_speed_controller'
        }.items()
    )

    # AS2 Motion Behaviors
    as2_behaviors_share = get_package_share_directory('as2_behaviors_motion')
    behaviors_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(as2_behaviors_share, 'launch', 'motion_behaviors_launch.py')),
        launch_arguments={
            'use_sim_time': 'true',
            'follow_path_plugin_name': 'follow_path_plugin_position',
            'go_to_plugin_name':      'go_to_plugin_position',
            'land_plugin_name':       'land_plugin_speed',
            'takeoff_plugin_name':    'takeoff_plugin_position',
            'follow_path_speed':      '1.0',
            'go_to_speed':            '1.0',
            'takeoff_height':         '3.0',
            'takeoff_speed':          '0.8',
            'land_speed':             '0.5',
        }.items()
    )

    # ----------------------
    # Helios app nodes (read /helios params YAML)
    # ----------------------
    # People detector
    people_detector = Node(
        package='helios_as2',
        executable='people_detector_node',
        name='people_detector_node',
        namespace=namespace,
        parameters=[params_file],
        output='screen'
    )

    # Reporter
    survivor_reporter = Node(
        package='helios_as2',
        executable='reporter_node',
        name='survivor_reporter',
        namespace=namespace,
        parameters=[params_file],
        output='screen'
    )

    # Mission patrol
    mission_patrol = Node(
        package='helios_as2',
        executable='mission_patrol',
        name='mission_patrol',
        namespace=namespace,
        parameters=[params_file],
        output='screen',
        arguments=['--autostart']
    )

    return LaunchDescription([
        declare_namespace,
        declare_params,
        sim,
        #bridge_clock,
        platform_launch,
        state_launch,
        controller_launch,
        behaviors_launch,
        people_detector,
        survivor_reporter,
        mission_patrol
    ])
