#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <nav_msgs/msg/path.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <geometry_msgs/msg/point.hpp>

#include "global_planner/tta_planner.hpp"

#include <queue>
#include <vector>
#include <algorithm>
#include <cstdint>
#include <cmath>

// 1) Subscribe to /map (nav_msgs/OccupancyGrid)
// 2) Process the occupancy grid to find two largest occupied components (walls)
// 3) Convert those components into vectors of BoundaryPoint
// 4) Run the TTA planner to compute the centerline
// 5) Publish the centerline as a nav_msgs/Path and a visualization_msgs/
class GlobalPlannerMapNode : public rclcpp::Node
{
public:
  GlobalPlannerMapNode()
  : Node("global_planner_map_node")
  {
    declare_parameter<std::string>("map_topic", "/map");
    declare_parameter<std::string>("frame_id", "odom");

    declare_parameter<int>("occ_value", 50);
    declare_parameter<int>("min_component_size", 200);
    declare_parameter<int>("stride", 2);
    declare_parameter<double>("close_centerline_distance", 0.0);

    declare_parameter<bool>("publish_wall_marker", true);
    declare_parameter<int>("max_wall_marker_points", 30000);

    const auto map_topic = get_parameter("map_topic").as_string();

    map_sub_ = create_subscription<nav_msgs::msg::OccupancyGrid>(
      map_topic,
      rclcpp::QoS(1).transient_local().reliable(),
      std::bind(&GlobalPlannerMapNode::mapCallback, this, std::placeholders::_1)
    );

    auto latched_qos = rclcpp::QoS(1).transient_local().reliable();

    path_pub_ = create_publisher<nav_msgs::msg::Path>("/global_centerline", latched_qos);
    center_marker_pub_ = create_publisher<visualization_msgs::msg::Marker>("/global_centerline_marker", latched_qos);
    wall_marker_pub_ = create_publisher<visualization_msgs::msg::Marker>("/map_walls_marker", latched_qos);

    RCLCPP_INFO(get_logger(), "Subscribed to /map and ready to compute centerline.");
  }

private:
  struct Cell
  {
    int x;
    int y;
  };

  inline bool inBounds(int x, int y, int width, int height) const
  {
    return x >= 0 && y >= 0 && x < width && y < height;
  }

  inline int index(int x, int y, int width) const
  {
    return y * width + x;
  }

  std::vector<std::vector<Cell>> findOccupiedComponents(
    const nav_msgs::msg::OccupancyGrid & grid,
    int occ_value,
    int min_component_size)
  {
    const int width = static_cast<int>(grid.info.width);
    const int height = static_cast<int>(grid.info.height);

    std::vector<uint8_t> visited(static_cast<size_t>(width) * static_cast<size_t>(height), 0);
    std::vector<std::vector<Cell>> components;

    const int dx[8] = {1, 1, 0, -1, -1, -1, 0, 1};
    const int dy[8] = {0, 1, 1, 1, 0, -1, -1, -1};

    for (int y = 0; y < height; ++y)
    {
      for (int x = 0; x < width; ++x)
      {
        const int idx = index(x, y, width);

        if (visited[idx]) {
          continue;
        }
        visited[idx] = 1;

        if (grid.data[idx] < occ_value) {
          continue;
        }

        std::vector<Cell> component;
        std::queue<Cell> q;
        q.push({x, y});

        while (!q.empty())
        {
          Cell c = q.front();
          q.pop();
          component.push_back(c);

          for (int k = 0; k < 8; ++k)
          {
            const int nx = c.x + dx[k];
            const int ny = c.y + dy[k];

            if (!inBounds(nx, ny, width, height)) {
              continue;
            }

            const int nidx = index(nx, ny, width);
            if (visited[nidx]) {
              continue;
            }
            visited[nidx] = 1;

            if (grid.data[nidx] >= occ_value) {
              q.push({nx, ny});
            }
          }
        }

        if (static_cast<int>(component.size()) >= min_component_size) {
          components.push_back(std::move(component));
        }
      }
    }

    std::sort(
      components.begin(), components.end(),
      [](const auto & a, const auto & b) {
        return a.size() > b.size();
      });

    return components;
  }

  std::vector<BoundaryPoint> componentToBoundaryPoints(
    const std::vector<Cell> & component,
    const nav_msgs::msg::OccupancyGrid & grid,
    int stride)
  {
    const double resolution = grid.info.resolution;
    const double origin_x = grid.info.origin.position.x;
    const double origin_y = grid.info.origin.position.y;

    std::vector<BoundaryPoint> points;
    points.reserve(component.size() / std::max(1, stride));

    int count = 0;
    for (const auto & c : component)
    {
      if (stride > 1 && (count++ % stride) != 0) {
        continue;
      }

      BoundaryPoint p;
      p.x = origin_x + static_cast<double>(c.x) * resolution;
      p.y = origin_y + static_cast<double>(c.y) * resolution;
      points.push_back(p);
    }

    return points;
  }

  void publishWallMarker(
    const nav_msgs::msg::OccupancyGrid & grid,
    const std::vector<Cell> & comp1,
    const std::vector<Cell> & comp2,
    const std::string & frame_id)
  {
    const int max_points = static_cast<int>(get_parameter("max_wall_marker_points").as_int());

    visualization_msgs::msg::Marker marker;
    marker.header.frame_id = frame_id;
    marker.header.stamp = now();
    marker.ns = "walls";
    marker.id = 0;
    marker.type = visualization_msgs::msg::Marker::POINTS;
    marker.action = visualization_msgs::msg::Marker::ADD;

    marker.scale.x = 0.05;
    marker.scale.y = 0.05;

    marker.color.a = 1.0;
    marker.color.r = 1.0;
    marker.color.g = 1.0;
    marker.color.b = 1.0;

    const double resolution = grid.info.resolution;
    const double origin_x = grid.info.origin.position.x;
    const double origin_y = grid.info.origin.position.y;

    auto addComponent = [&](const std::vector<Cell> & comp)
    {
      for (const auto & c : comp)
      {
        if (static_cast<int>(marker.points.size()) >= max_points) {
          break;
        }

        geometry_msgs::msg::Point p;
        p.x = origin_x + static_cast<double>(c.x) * resolution;
        p.y = origin_y + static_cast<double>(c.y) * resolution;
        p.z = 0.0;
        marker.points.push_back(p);
      }
    };

    addComponent(comp1);
    addComponent(comp2);

    wall_marker_pub_->publish(marker);
  }

  void publishCenterline(
    const std::vector<BoundaryPoint> & centerline,
    const std::string & frame_id)
  {
    std::vector<BoundaryPoint> published_centerline = centerline;
    const double close_distance = get_parameter("close_centerline_distance").as_double();

    if (close_distance > 0.0 && published_centerline.size() > 2)
    {
      const auto & first = published_centerline.front();
      const auto & last = published_centerline.back();
      const double dx = last.x - first.x;
      const double dy = last.y - first.y;
      const double endpoint_distance = std::hypot(dx, dy);

      if (endpoint_distance <= close_distance) {
        published_centerline.push_back(first);
      }
    }

    nav_msgs::msg::Path path;
    path.header.frame_id = frame_id;
    path.header.stamp = now();

    for (const auto & p : published_centerline)
    {
      geometry_msgs::msg::PoseStamped pose;
      pose.header = path.header;
      pose.pose.position.x = p.x;
      pose.pose.position.y = p.y;
      pose.pose.position.z = 0.0;
      pose.pose.orientation.w = 1.0;
      path.poses.push_back(pose);
    }

    path_pub_->publish(path);

    visualization_msgs::msg::Marker marker;
    marker.header.frame_id = frame_id;
    marker.header.stamp = now();
    marker.ns = "centerline";
    marker.id = 0;
    marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
    marker.action = visualization_msgs::msg::Marker::ADD;

    marker.scale.x = 0.10;

    marker.color.a = 1.0;
    marker.color.r = 0.0;
    marker.color.g = 1.0;
    marker.color.b = 0.0;

    for (const auto & p : published_centerline)
    {
      geometry_msgs::msg::Point gp;
      gp.x = p.x;
      gp.y = p.y;
      gp.z = 0.0;
      marker.points.push_back(gp);
    }

    center_marker_pub_->publish(marker);
  }

  void mapCallback(const nav_msgs::msg::OccupancyGrid::SharedPtr msg)
  {
    if (computed) {
      return;
    }

    const std::string frame_id = get_parameter("frame_id").as_string();
    const int occ_value = static_cast<int>(get_parameter("occ_value").as_int());
    const int min_component_size = static_cast<int>(get_parameter("min_component_size").as_int());
    const int stride = std::max(1, static_cast<int>(get_parameter("stride").as_int()));
    const bool publish_wall_marker = get_parameter("publish_wall_marker").as_bool();

    // 1) Find occupied blobs
    auto components = findOccupiedComponents(*msg, occ_value, min_component_size);

    if (components.size() < 2)
    {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 2000,
        "Only found %zu occupied components >= %d cells. Try lowering min_component_size.",
        components.size(), min_component_size);
      return;
    }

    // 2) Take the two largest blobs as the two walls
    const auto & wall1 = components[0];
    const auto & wall2 = components[1];

    if (publish_wall_marker) {
      publishWallMarker(*msg, wall1, wall2, frame_id);
    }

    // 3) Convert them into BoundaryPoint vectors
    std::vector<BoundaryPoint> boundary1 = componentToBoundaryPoints(wall1, *msg, stride);
    std::vector<BoundaryPoint> boundary2 = componentToBoundaryPoints(wall2, *msg, stride);

    // 4) Run your existing TTA planner
    std::vector<BoundaryPoint> centerline;
    bool success = planner_.computeCenterline(boundary1, boundary2, centerline);

    if (!success || centerline.empty())
    {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 2000,
        "computeCenterline failed or produced empty output.");
      return;
    }

    // 5) Publish result
    publishCenterline(centerline, frame_id);

    computed = true;

    RCLCPP_INFO_THROTTLE(
      get_logger(), *get_clock(), 2000,
      "Wall1: %zu points, Wall2: %zu points, Centerline: %zu points",
      boundary1.size(), boundary2.size(), centerline.size());
  }

  rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr map_sub_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr center_marker_pub_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr wall_marker_pub_;

  TTAPlanner planner_;

  bool computed = false;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<GlobalPlannerMapNode>());
  rclcpp::shutdown();
  return 0;
}
