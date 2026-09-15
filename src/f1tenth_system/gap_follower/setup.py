from setuptools import setup
from setuptools import find_packages
import os
from glob import glob

package_name = 'gap_follower'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        (
            'share/ament_index/resource_index/packages',
            ['resource/' + package_name]
        ),
        (
            'share/' + package_name,
            ['package.xml']
        ),
        (
            os.path.join('share', package_name, 'launch'),
            glob('launch/*.launch.py')
        ),
        (
            os.path.join('share', package_name, 'config'),
            glob('config/*.yaml')
        ),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='team',
    maintainer_email='noreply@example.com',
    description='Gap follower local planner package',
    license='TODO',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'gap_follower_node = gap_follower.gap_follower:main',
            'path_follower_node = gap_follower.path_follower:main',
        ],
    },
)
