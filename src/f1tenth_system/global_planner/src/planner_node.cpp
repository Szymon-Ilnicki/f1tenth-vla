#include <memory>
#include <vector>

#include "global_planner/tta_planner.hpp"
#include "rclcpp/rclcpp.hpp"

#include "sensor_msgs/msg/laser_scan.hpp"
#include "nav_msgs/msg/odometry.hpp"

#include <nav_msgs/msg/path.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include "ackermann_msgs/msg/ackermann_drive_stamped.hpp"
#include <chrono>

#include <cmath>
#include <string>

//class definition
class GlobalPlannerNode : public rclcpp::Node
{
public:
  //contructor
  GlobalPlannerNode()
  : Node("global_planner_node")
  {
    //parameters
    this->declare_parameter<double>("default_speed", 0.0);
    this->declare_parameter<double>("max_steering_angle", 0.4);
    this->declare_parameter<std::string>("frame_id", "map");
    
    this->declare_parameter<std::string>("test_track", "circle");
    this->declare_parameter<int>("test_points", 120);
    this->declare_parameter<double>("track_half_width", 1.0);

    test_track_ = parseTestTrack(this->get_parameter("test_track").as_string());
    test_points_ = this->get_parameter("test_points").as_int();
    track_half_width_ = this->get_parameter("track_half_width").as_double();

    default_speed_ = this->get_parameter("default_speed").as_double();
    max_steering_angle_ = this->get_parameter("max_steering_angle").as_double();
    frame_id_ = this->get_parameter("frame_id").as_string();

    //subscriptions and publishers
    odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
      "/odom", 10,
      [this](nav_msgs::msg::Odometry::SharedPtr m) {odom_ = m; step();}
    );

    scan_sub_ = create_subscription<sensor_msgs::msg::LaserScan>(
      "/scan", 10,
      [this](sensor_msgs::msg::LaserScan::SharedPtr m) {scan_ = m; step();}
    );

    drive_pub_ = create_publisher<ackermann_msgs::msg::AckermannDriveStamped>("/drive", 10);


    //path publisher with latching
    rclcpp::QoS qos(rclcpp::KeepLast(1));
    qos.transient_local();   // latch last message
    qos.reliable();

    path_pub_ = create_publisher<nav_msgs::msg::Path>(
      "/global_centerline", qos
    );

    //timer for step function
    timer_ = create_wall_timer(
      std::chrono::milliseconds(500),
      std::bind(&GlobalPlannerNode::step, this)
    );

  }

private:

  enum class TestTrack
  {
    Straight,
    Circle,
    Oval,
  };

  TestTrack test_track_{TestTrack::Circle};
  int test_points_{120};
  double track_half_width_{1.0};

  static TestTrack parseTestTrack(const std::string & track_str)
  {
    if (track_str == "straight") {
      return TestTrack::Straight;
    }
    if (track_str == "circle") {
      return TestTrack::Circle;
    }
    if (track_str == "oval") {
      return TestTrack::Oval;
    }
    return TestTrack::Circle; // default
  }

  void generateTestBoundaries(std::vector<BoundaryPoint> & left, std::vector<BoundaryPoint> & right) const
  {
    left.clear();
    right.clear();
    left.reserve(static_cast<size_t>(test_points_));
    right.reserve(static_cast<size_t>(test_points_));

    const int n = std::max(20, test_points_);
    const double w = std::max(0.05, track_half_width_); // track width

    auto push_lr = [&](double cx, double cy, double nx, double ny) 
    {
      left.push_back({cx + w * nx, cy + w * ny});
      right.push_back({cx - w * nx, cy - w * ny});
    };

    if(test_track_ == TestTrack::Straight) {
     for (int i  = 0; i < n; ++i)
     {
      double t = static_cast<double>(i) / (n-1);
      double cx = 0.0;
      double cy = 10.0 * t;
      push_lr(cx, cy, -1.0, 0.0);
     }
     return;
    }
    if(test_track_ == TestTrack::Circle) 
    {
      const double R = 5.0;
       for (int i = 0; i < n; ++i) {
        double theta = 2.0 * M_PI * static_cast<double>(i) / n;
        double cx = R * cos(theta);
        double cy = R * sin(theta);
        double nx = cos(theta);
        double ny = sin(theta);

        push_lr(cx, cy, nx, ny);
      }
      return;
    }
    if (test_track_ == TestTrack::Oval) 
    {
      const double a = 7.0; // semi-major axis
      const double b = 4.0; // semi-minor axis

      for (int i = 0; i < n; ++i) {
        double t = 2.0 * M_PI * static_cast<double>(i) / n;
        double cx = a * std::cos(t);
        double cy = b * std::sin(t);

        double tx = -a * std::sin(t);
        double ty =  b * std::cos(t);

        double nx = -ty;
        double ny =  tx;
        double norm = std::hypot(nx, ny);
        if (norm > 1e-9) {
          nx /= norm;
          ny /= norm;
        }
        push_lr(cx, cy, nx, ny);
      }
      return;
    }
  }
  
  //main step function
  void step()
  {
    //const bool have_sensors = (odom_ && scan_);

    std::vector<BoundaryPoint> left;
    std::vector<BoundaryPoint> right;
    generateTestBoundaries(left, right);

    std::vector<BoundaryPoint> centerline;

    //calls planner to compute centerline
    if (planner_.computeCenterline(left, right, centerline)) {
      //convert centerline to path message and publish
      nav_msgs::msg::Path path_msg;
      path_msg.header.stamp = now();
      path_msg.header.frame_id = frame_id_;

      for (const auto & pt : centerline) {
        geometry_msgs::msg::PoseStamped pose;
        pose.header.stamp = now();
        pose.header.frame_id = frame_id_;
        pose.pose.position.x = pt.x;
        pose.pose.position.y = pt.y;
        pose.pose.position.z = 0.0;
        pose.pose.orientation.w = 1.0;
        path_msg.poses.push_back(pose);
      }
      path_pub_->publish(path_msg);

      RCLCPP_INFO_THROTTLE(
        get_logger(), *get_clock(), 2000,
        "Published global centerline with %zu points (first = %.2f, %.2f)",
        centerline.size(), centerline[0].x, centerline[0].y
      );
    }
    else
    {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "Failed to compute centerline");

    }

    ackermann_msgs::msg::AckermannDriveStamped cmd;
    cmd.header.stamp = now();
    //placeholder to stop car doing anything
    cmd.drive.steering_angle = 0.0;
    cmd.drive.speed = 0.0;
    drive_pub_->publish(cmd);
  }

  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_sub_;
  rclcpp::Publisher<ackermann_msgs::msg::AckermannDriveStamped>::SharedPtr drive_pub_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;

  rclcpp::TimerBase::SharedPtr timer_;

  nav_msgs::msg::Odometry::SharedPtr odom_;
  sensor_msgs::msg::LaserScan::SharedPtr scan_;


  TTAPlanner planner_;

  double default_speed_{0.0};
  double max_steering_angle_{0.4};
  std::string frame_id_{"map"};

};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<GlobalPlannerNode>());
  rclcpp::shutdown();
  return 0;
}