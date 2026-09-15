

global_planner/
├── config/
│   └── planner_params.yaml          # Parameters for planner tuning
│
├── include/global_planner/
│   └── tta_planner.hpp              # TTA planner class definition
│
├── launch/
│   ├── global_planner_launch.py     # ROS2 launch file
│   └── README.md                    # Launch usage notes
│
├── maps/
│   ├── Austin_map.png
│   ├── Austin_map.yaml
│   ├── Shanghai_map.png
│   └── Shanghai_map.yaml            # Static maps for testing
│
├── src/
│   ├── global_planner_map_node.cpp  # Node using occupancy grid/map
│   ├── planner_node.cpp             # Main planner node
│   └── tta_planner.cpp              # TTA planner implementation
│
├── test/
│   └── test_tta_planner.cpp         # Unit tests (gtest)
│
├── CMakeLists.txt
├── package.xml
└── README.md
