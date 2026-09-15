from setuptools import setup

package_name = "vehicle_test"

setup(
    name=package_name,
    version="0.0.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="ethan",
    maintainer_email="ethan@example.com",
    description="Simple vehicle drive test node",
    license="MIT",
    entry_points={
        "console_scripts": [
            "vehicle_test = vehicle_test.test_drive:main",
        ],
    },
)
