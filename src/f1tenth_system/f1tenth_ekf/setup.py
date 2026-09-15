from setuptools import setup

package_name = 'f1tenth_ekf'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/ekf.launch.py']),
        ('share/' + package_name + '/config', ['config/ekf.yaml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='lewis',
    maintainer_email='lm2075@hw.ac.uk',
    description='F1TENTH EKF package wrapping robot_localization ekf_node',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'ekf_node = f1tenth_ekf.ekf_node:main',
        ],
    },
)
