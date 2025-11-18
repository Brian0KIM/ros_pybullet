// student ID: 2021105575
// name: 김성민
//step1_practice.cpp 를 참고하여 구현하세요

#include <rclcpp/rclcpp.hpp>

// MoveIt 2
#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include <moveit/move_group_interface/move_group_interface.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <cmath>
#include <iostream>

#include <thread>
#include <chrono>

// Location class for 2D positions
class Location {
public:
  double x;
  double y;
  Location() = default;
  Location(double x_value, double y_value) : x(x_value), y(y_value) {}
};

// ObjectKind identifies shape used for grasp parameterization
enum class ObjectKind {
  Box,
  Cylinder,
  Triangle
};

// Object specification for pick-and-place
struct ObjectSpec {
  std::string name;
  ObjectKind kind;
  // Dimensions
  double size_x;   // used for Box/Triangle
  double size_y;   // used for Box/Triangle
  double height;   // used for all objects
  double radius;   // used for Cylinder
  // Poses
  Location pick_location;
  Location place_location;
  // Preferred yaw during grasp if object requires it
  double grasp_yaw;
  // Preferred yaw during placement (may differ from grasp yaw)
  double place_yaw;
};

// Convert simple list to geometry pose
geometry_msgs::msg::Pose list_to_pose(double x, double y, double z,
                                      double roll, double pitch, double yaw) {
  geometry_msgs::msg::Pose pose;
  tf2::Quaternion q;
  q.setRPY(roll, pitch, yaw);
  pose.position.x = x;
  pose.position.y = y;
  pose.position.z = z;
  pose.orientation = tf2::toMsg(q);
  return pose;
}

// Execute a joint-space plan to a pose target
void go_to_pose_goal(moveit::planning_interface::MoveGroupInterface &move_group_interface,
                     geometry_msgs::msg::Pose &target_pose) {
  move_group_interface.setPoseTarget(target_pose);
  moveit::planning_interface::MoveGroupInterface::Plan my_plan;
  auto planning_result = move_group_interface.plan(my_plan);
  bool success = (planning_result == moveit::planning_interface::MoveItErrorCode::SUCCESS);
  if (success) {
    move_group_interface.execute(my_plan);
  }
}

// Compute cartesian path along waypoints and execute
void execute_waypoints(moveit::planning_interface::MoveGroupInterface &move_group_interface,
                       rclcpp::Node::SharedPtr node,
                       const std::vector<geometry_msgs::msg::Pose> &waypoints) {
  // Slow and smooth Cartesian execution
  move_group_interface.setMaxVelocityScalingFactor(0.15);
  move_group_interface.setMaxAccelerationScalingFactor(0.10);
  moveit_msgs::msg::RobotTrajectory trajectory;
  double fraction = move_group_interface.computeCartesianPath(waypoints, 0.003, 0.0, trajectory);
  (void)fraction; // keep consistent with -Wall
  moveit::planning_interface::MoveGroupInterface::Plan cartesian_plan;
  cartesian_plan.trajectory_ = trajectory;
  move_group_interface.execute(cartesian_plan);
  rclcpp::sleep_for(std::chrono::milliseconds(200));
}

// Gripper helpers
void set_gripper(moveit::planning_interface::MoveGroupInterface &gripper_interface, double value) {
  moveit::planning_interface::MoveGroupInterface::Plan my_plan;
  // Slow down gripper actuation slightly for stable contacts
  gripper_interface.setMaxVelocityScalingFactor(0.30);
  gripper_interface.setMaxAccelerationScalingFactor(0.30);
  std::vector<double> joint_group_positions = gripper_interface.getCurrentJointValues();
  if (joint_group_positions.empty()) {
    return;
  }
  joint_group_positions[0] = value;
  joint_group_positions[1] = 0.0;
  gripper_interface.setJointValueTarget(joint_group_positions);
  auto planning_result = gripper_interface.plan(my_plan);
  bool success = (planning_result == moveit::planning_interface::MoveItErrorCode::SUCCESS);
  if (success) {
    gripper_interface.execute(my_plan);
    rclcpp::sleep_for(std::chrono::milliseconds(150));
  }
}

void open_gripper(moveit::planning_interface::MoveGroupInterface &gripper_interface) {
  set_gripper(gripper_interface, 0.07);
}

void close_gripper(moveit::planning_interface::MoveGroupInterface &gripper_interface, double value) {
  set_gripper(gripper_interface, value);
}

// Initial arm posture for safe start
void move_to_initial_pose(moveit::planning_interface::MoveGroupInterface &arm_interface) {
  moveit::planning_interface::MoveGroupInterface::Plan my_plan;
  std::vector<double> joint_group_positions = arm_interface.getCurrentJointValues();
  if (joint_group_positions.size() >= 6) {
    joint_group_positions = {0, -2.03, 1.58, -1.19, -1.58, 0.78};
    arm_interface.setJointValueTarget(joint_group_positions);
    auto planning_result = arm_interface.plan(my_plan);
    bool success = (planning_result == moveit::planning_interface::MoveItErrorCode::SUCCESS);
    if (success) {
      arm_interface.execute(my_plan);
    }
  }
}

// Compute gripper closing value from object dims
double compute_grasp_width(const ObjectSpec &obj) {
  // Special tuning for triangle mesh based on reference behavior
  if (obj.kind == ObjectKind::Triangle) {
    return 0.0175;
  }
  double effective_width = 0.0;
  if (obj.kind == ObjectKind::Cylinder) {
    effective_width = 2.0 * obj.radius;
  } else {
    effective_width = std::min(obj.size_x, obj.size_y);
  }
  // Provide a small margin to ensure contact without crushing
  double value = std::max(0.0, effective_width * 0.5 - 0.001);
  // Clamp to gripper limits
  if (value > 0.04) value = 0.04;
  return value;
}

// Perform pick sequence
void pick_object(rclcpp::Node::SharedPtr node,
                 moveit::planning_interface::MoveGroupInterface &arm_interface,
                 moveit::planning_interface::MoveGroupInterface &gripper_interface,
                 const ObjectSpec &obj) {
  const double approach_height = 0.40;
  const double tool_offset = 0.185; // wrist to fingertips offset
  const double grasp_z = obj.height + tool_offset;
  const double roll = M_PI;
  const double pitch = 0.0;
  const double yaw = obj.grasp_yaw;

  geometry_msgs::msg::Pose above_pick = list_to_pose(obj.pick_location.x, obj.pick_location.y,
                                                     approach_height, roll, pitch, yaw);
  geometry_msgs::msg::Pose touch_pick = list_to_pose(obj.pick_location.x, obj.pick_location.y,
                                                     grasp_z, roll, pitch, yaw);

  std::vector<geometry_msgs::msg::Pose> wp1;
  wp1.push_back(above_pick);
  execute_waypoints(arm_interface, node, wp1);
  open_gripper(gripper_interface);

  std::vector<geometry_msgs::msg::Pose> wp2;
  wp2.push_back(touch_pick);
  execute_waypoints(arm_interface, node, wp2);
  close_gripper(gripper_interface, compute_grasp_width(obj));

  std::vector<geometry_msgs::msg::Pose> wp3;
  wp3.push_back(above_pick);
  execute_waypoints(arm_interface, node, wp3);
}

// Perform place sequence
void place_object(rclcpp::Node::SharedPtr node,
                  moveit::planning_interface::MoveGroupInterface &arm_interface,
                  moveit::planning_interface::MoveGroupInterface &gripper_interface,
                  const ObjectSpec &obj) {
  const double approach_height = 0.40;
  const double tool_offset = 0.185;
  // Target height tuned to drop objects cleanly into case cavities
  const double place_z = 0.125 + tool_offset;
  const double roll = M_PI;
  const double pitch = 0.0;
  const double yaw = obj.place_yaw;

  geometry_msgs::msg::Pose above_place = list_to_pose(obj.place_location.x, obj.place_location.y,
                                                      approach_height, roll, pitch, yaw);
  geometry_msgs::msg::Pose touch_place = list_to_pose(obj.place_location.x, obj.place_location.y,
                                                      place_z, roll, pitch, yaw);

  std::vector<geometry_msgs::msg::Pose> wp1;
  wp1.push_back(above_place);
  wp1.push_back(touch_place);
  execute_waypoints(arm_interface, node, wp1);
  open_gripper(gripper_interface);

  std::vector<geometry_msgs::msg::Pose> wp2;
  wp2.push_back(above_place);
  execute_waypoints(arm_interface, node, wp2);
}

class Step2HomeworkNode : public rclcpp::Node {
public:
  Step2HomeworkNode(const rclcpp::NodeOptions &options = rclcpp::NodeOptions())
      : Node("step2_hw_node", options) {
    RCLCPP_INFO(this->get_logger(), "Step2 Homework Node started.");
  }

  void run() {
    auto node_ptr = this->shared_from_this();
    moveit::planning_interface::MoveGroupInterface arm(node_ptr, "manipulator");
    moveit::planning_interface::MoveGroupInterface gripper(node_ptr, "gripper");
    moveit::planning_interface::PlanningSceneInterface planning_scene_interface;

    arm.setPlanningTime(20.0);
    rclcpp::sleep_for(std::chrono::seconds(1));

    // Ground plane for collision safety
    moveit_msgs::msg::CollisionObject ground_plane;
    ground_plane.header.frame_id = arm.getPlanningFrame();
    ground_plane.id = "ground_plane";

    shape_msgs::msg::SolidPrimitive plane_primitive;
    plane_primitive.type = plane_primitive.BOX;
    plane_primitive.dimensions = {4.0, 4.0, 0.01};

    geometry_msgs::msg::Pose plane_pose;
    plane_pose.orientation.w = 1.0;
    plane_pose.position.z = -0.005;

    ground_plane.primitives.push_back(plane_primitive);
    ground_plane.primitive_poses.push_back(plane_pose);
    ground_plane.operation = ground_plane.ADD;
    planning_scene_interface.applyCollisionObjects({ground_plane});

    // Start posture
    move_to_initial_pose(arm);
    close_gripper(gripper, 0.0);

    // Preferred yaw: align gripper diagonally to the table for robust grasps
    const double default_yaw = -M_PI / 4.0;

    // Prepare objects based on spawned scene
    std::vector<ObjectSpec> objects;
    // Note: place_yaw equals grasp_yaw unless otherwise specified
    objects.push_back({"box1", ObjectKind::Box, 0.05, 0.05, 0.06, 0.0,
                       Location(0.4, 0.1), Location(-0.15, 0.45), default_yaw, default_yaw});
    objects.push_back({"box2", ObjectKind::Box, 0.05, 0.05, 0.06, 0.0,
                       Location(0.4, 0.0), Location(-0.15, 0.55), default_yaw, default_yaw});
    // Thin rectangle: rotate additional 45 degrees on placement
    objects.push_back({"box3", ObjectKind::Box, 0.05, 0.025, 0.06, 0.0,
                       Location(0.4, -0.1), Location(0.15, 0.45), default_yaw, M_PI/4.0});
    objects.push_back({"box4", ObjectKind::Box, 0.05, 0.05, 0.07, 0.0,
                       Location(0.5, -0.1), Location(-0.05, 0.55), default_yaw, default_yaw});
    objects.push_back({"box5", ObjectKind::Box, 0.05, 0.05, 0.08, 0.0,
                       Location(0.5, 0.1), Location(0.05, 0.55), default_yaw, default_yaw});
    objects.push_back({"box6", ObjectKind::Box, 0.05, 0.05, 0.09, 0.0,
                       Location(0.6, 0.0), Location(0.15, 0.55), default_yaw, default_yaw});
    objects.push_back({"cylinder", ObjectKind::Cylinder, 0.0, 0.0, 0.06, 0.025,
                       Location(0.5, 0.0), Location(-0.05, 0.45), default_yaw, default_yaw});
    // Triangle: pick last, adjust grasp yaw like the reference (diagonal pinch)
    objects.push_back({"triangle", ObjectKind::Triangle, 0.05, 0.05, 0.06, 0.0,
                       Location(0.6 + 0.00625, 0.1), Location(0.05 + 0.00625, 0.45), M_PI/4.0, -M_PI/4.0});

    // Execute pick-and-place for all objects
    for (const auto &obj : objects) {
      pick_object(node_ptr, arm, gripper, obj);
      place_object(node_ptr, arm, gripper, obj);
    }
  }
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<Step2HomeworkNode>();

  std::thread spinner([node]() { rclcpp::spin(node); });
  spinner.detach();

  node->run();
  rclcpp::shutdown();

  return 0;
}