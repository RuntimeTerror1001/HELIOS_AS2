# setup.py
from setuptools import setup, find_packages
from glob import glob
import os

package_name = 'helios_as2'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(include=[package_name]),
    data_files=[
        ('share/ament_index/resource_index/packages', [f'resource/{package_name}']),
        (f'share/{package_name}', ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py') + glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), [
            'config/ekf_odom.yaml',
            'config/navsat.yaml',
            'config/rtabmap_rgbd.yaml',
            'config/helios_sim.json',
            'config/params.yaml'
        ]),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='redpaladin',
    maintainer_email='parthraj1001@gmail.com',
    description='Helios AS2 main package',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            # your files under helios_as2/
            'mission_patrol = helios_as2.mission_patrol:main',
            'reporter_node  = helios_as2.reporter_node:main',
            'people_detector_node = helios_as2.people_detector_node:main',
        ],
    },
)
