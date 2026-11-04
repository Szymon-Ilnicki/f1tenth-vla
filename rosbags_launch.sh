#!/bin/bash
# save as ~/rosbags_launch.sh
# chmod +x ~/rosbags_launch.sh

# === CONFIG ===
ROS_SETUP="/opt/ros/humble/setup.bash"
WORKSPACE_SETUP="install/setup.bash"
CAR_LAUNCH="src/f1tenth_system/f1tenth_stack/launch/car_cam.launch.py"
BAG_RECORDER="src/f1tenth_system/f1tenth_stack/f1tenth_stack/bag_recorder_node.py"   # path to the Python node (adjust if needed)
# ===========================

# Source ROS
source "$ROS_SETUP" 2>/dev/null || true
[ -f "$WORKSPACE_SETUP" ] && source "$WORKSPACE_SETUP"

# Wake lock so Termux is less likely to be killed
termux-wake-lock 2>/dev/null || true

echo "Starting camera stack..."
ros2 launch "$CAR_LAUNCH" &
LAUNCH_PID=$!
echo "Camera stack PID: $LAUNCH_PID"

# Give the launch a moment
sleep 3
ros2 topic pub --once /task_in std_msgs/msg/String "{data: 'drive forward slowly'}"

echo "Starting bag recorder..."
python3 "$BAG_RECORDER" &
RECORDER_PID=$!
echo "Bag recorder PID: $RECORDER_PID"

echo ""
echo "Both running. Use joystick buttons to start/stop bags."
echo "Press Ctrl+C to stop everything."

# Wait and clean up on exit
trap "kill $LAUNCH_PID $RECORDER_PID 2>/dev/null; termux-wake-unlock 2>/dev/null; exit" INT TERM
wait
