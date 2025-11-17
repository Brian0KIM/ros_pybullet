import pybullet as p
import pybullet_data
import time
import numpy as np

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.8)

p.loadURDF("plane.urdf")
table = p.loadURDF("table/table.urdf", [0.5, 0, 0], useFixedBase=True)
robot = p.loadURDF("franka_panda/panda.urdf", basePosition=[0.0, 0, 0.66], useFixedBase=True)

box_6 = p.loadURDF("models/box6.xacro", basePosition=[0.3, -0.1, 0.66])
box_5 = p.loadURDF("models/box5.xacro", basePosition=[0.3, 0.0, 0.66])
box_4 = p.loadURDF("models/box4.xacro", basePosition=[0.3, 0.1, 0.66])

box_3 = p.loadURDF("models/box3.xacro", basePosition=[0.4, -0.1, 0.66])
cylinder_0 = p.loadURDF("models/cylinder1.xacro", basePosition=[0.4, 0.0, 0.66])
box_2 = p.loadURDF("models/box2.xacro", basePosition=[0.4, 0.1, 0.66])

box_0 = p.loadURDF("models/box.xacro", basePosition=[0.5, 0.0, 0.66])
triangle = p.loadURDF("models/triangle.xacro", basePosition=[0.5, 0.1, 0.66])

case_collision = p.createCollisionShape(
    shapeType=p.GEOM_MESH,
    fileName="models/case.obj", 
    flags=p.GEOM_FORCE_CONCAVE_TRIMESH,
    meshScale=[1, 1, 1]
)

case_visual = p.createVisualShape(
    shapeType=p.GEOM_MESH,
    fileName="models/case.obj",
    meshScale=[1, 1, 1]
)

case = p.createMultiBody(
    baseMass=0.05,
    baseCollisionShapeIndex=case_collision,
    baseVisualShapeIndex=case_visual,
    basePosition=[0.0, 0.3, 0.64],
    baseOrientation=p.getQuaternionFromEuler([0, 0, 3.14159])
)


p.changeVisualShape(box_6, -1, rgbaColor=[1, 0, 1, 1])  # box
p.changeVisualShape(box_5, -1, rgbaColor=[1, 0, 0, 1])  # box2
p.changeVisualShape(box_4, -1, rgbaColor=[1, 0, 0, 1])  # box3

p.changeVisualShape(box_3, -1, rgbaColor=[1, 0, 0, 1])  # box4
p.changeVisualShape(cylinder_0, -1, rgbaColor=[0, 0, 1, 1])  # box5
p.changeVisualShape(box_2, -1, rgbaColor=[1, 0, 0, 1])  # box6

p.changeVisualShape(box_0, -1, rgbaColor=[1, 0, 0, 1])  # box5
p.changeVisualShape(triangle, -1, rgbaColor=[1, 0, 1, 1])  # box6

p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=45, cameraPitch=-30, cameraTargetPosition=[0.3, 0, 0.5])

## 기본적인 Scene 코드 이후 부분을 제공된 Task를 수행하도록 구현하세요 ##

# ----------------------------
# Helper functions (IK/Control)
# ----------------------------

def step_for_seconds(seconds: float, hz: int = 240) -> None:
    """Step the physics for given seconds."""
    steps = int(seconds * hz)
    for _ in range(steps):
        p.stepSimulation()
        time.sleep(1.0 / hz)


def get_panda_indices(robot_id: int):
    """Return arm and finger joint indices and an end-effector link index."""
    num_joints = p.getNumJoints(robot_id)
    revolute_indices = []
    prismatic_indices = []
    ee_index_candidates = []
    for j in range(num_joints):
        info = p.getJointInfo(robot_id, j)
        j_type = info[2]
        link_name = info[12].decode("utf-8")
        joint_name = info[1].decode("utf-8")
        if j_type == p.JOINT_REVOLUTE:
            revolute_indices.append(j)
        if j_type == p.JOINT_PRISMATIC:
            prismatic_indices.append(j)
        # Try to detect an end-effector-ish link
        if any(key in link_name.lower() for key in ["hand", "grip", "grasp", "link7"]):
            ee_index_candidates.append(j)

    # First 7 revolute joints are Panda arm joints
    arm_indices = revolute_indices[:7]

    # First two prismatic joints are Panda fingers
    finger_indices = prismatic_indices[:2] if len(prismatic_indices) >= 2 else prismatic_indices

    # Heuristic for EE link index (works for default PyBullet Panda)
    if ee_index_candidates:
        ee_index = ee_index_candidates[-1]
    else:
        ee_index = 11 if p.getNumJoints(robot_id) > 11 else (p.getNumJoints(robot_id) - 1)

    return arm_indices, finger_indices, ee_index


def set_arm_positions(robot_id: int, arm_indices, target_positions, max_force: float = 200.0):
    """Position control of arm joints."""
    p.setJointMotorControlArray(
        bodyUniqueId=robot_id,
        jointIndices=arm_indices,
        controlMode=p.POSITION_CONTROL,
        targetPositions=target_positions[: len(arm_indices)],
        forces=[max_force] * len(arm_indices),
    )


def set_gripper(robot_id: int, finger_indices, width: float, force: float = 40.0):
    """Open/close Panda gripper by setting both prismatic finger joints to same value.
    width is the target position per each finger (0.0 ~ 0.04).
    """
    if not finger_indices:
        return
    tgt = max(0.0, min(0.04, width))
    for j in finger_indices:
        p.setJointMotorControl2(
            bodyUniqueId=robot_id,
            jointIndex=j,
            controlMode=p.POSITION_CONTROL,
            targetPosition=tgt,
            force=force,
        )


def move_ee_linear(robot_id: int, arm_indices, ee_index: int, start_pos, end_pos, orn, duration: float = 1.2, steps: int = 120):
    """Move end-effector linearly by IK along straight path from start_pos to end_pos."""
    for i in range(steps):
        alpha = (i + 1) / steps
        tgt = [
            start_pos[0] * (1 - alpha) + end_pos[0] * alpha,
            start_pos[1] * (1 - alpha) + end_pos[1] * alpha,
            start_pos[2] * (1 - alpha) + end_pos[2] * alpha,
        ]
        # Use default IK
        q = p.calculateInverseKinematics(robot_id, ee_index, tgt, orn)
        set_arm_positions(robot_id, arm_indices, q)
        p.stepSimulation()
        time.sleep(duration / steps)


def get_ee_world_pose(robot_id: int, ee_index: int):
    """Return current world pose of end-effector link."""
    ls = p.getLinkState(robot_id, ee_index)
    return ls[4], ls[5]


def go_home(robot_id: int, arm_indices):
    """Move to a simple home configuration that keeps wrist above table."""
    # A conservative, reachable posture for default Panda
    home_q = [0.0, -0.5, 0.0, -2.2, 0.0, 1.7, 0.8]
    set_arm_positions(robot_id, arm_indices, home_q)
    step_for_seconds(1.2)


def freeze_body_to_world(body_id: int):
    """Fix given body to the world using a fixed constraint, to avoid wobbling."""
    pos, orn = p.getBasePositionAndOrientation(body_id)
    p.createConstraint(
        parentBodyUniqueId=body_id,
        parentLinkIndex=-1,
        childBodyUniqueId=-1,
        childLinkIndex=-1,
        jointType=p.JOINT_FIXED,
        jointAxis=[0, 0, 0],
        parentFramePosition=[0, 0, 0],
        childFramePosition=pos,
        parentFrameOrientation=[0, 0, 0, 1],
        childFrameOrientation=orn,
    )
    # Let the solver settle
    step_for_seconds(0.25)


# ----------------------------
# Main pick-and-place routine
# ----------------------------

# Mark TODO: Fix case wobble (allowed)
freeze_body_to_world(case)

# Acquire indices and EE link
arm_joint_indices, finger_joint_indices, ee_link_index = get_panda_indices(robot)

# Orientation: gripper facing down
downward_orn = p.getQuaternionFromEuler([np.pi, 0.0, 0.0])

# Prepare
go_home(robot, arm_joint_indices)
set_gripper(robot, finger_joint_indices, width=0.038)  # open
step_for_seconds(0.3)

# ----------------------------
# Case slot layout (adaptive)
# ----------------------------
# NOTE: The case mesh (case.obj/case.dae) does not encode slot coordinates as parameters.
#       We therefore compute them from the case's current size (AABB) and pose, making
#       the target positions robust to case placement and rotation.

def compute_case_slot_world_xy(case_id: int, cols: int = 4, rows: int = 2,
                               margin_x_ratio: float = 0.22, margin_y_ratio: float = 0.28):
    """Compute world XY of slot centers as a grid inside case footprint.

    The grid is defined in the case local frame using AABB size, then transformed
    to world coordinates with the case base pose.
    """
    (aabb_min_x, aabb_min_y, aabb_min_z), (aabb_max_x, aabb_max_y, aabb_max_z) = p.getAABB(case_id)
    size_x = max(1e-6, aabb_max_x - aabb_min_x)
    size_y = max(1e-6, aabb_max_y - aabb_min_y)
    half_x = size_x * 0.5
    half_y = size_y * 0.5

    # Inner rectangle margins (tunable to match actual holes)
    margin_x = size_x * margin_x_ratio
    margin_y = size_y * margin_y_ratio

    inner_min_x = -half_x + margin_x
    inner_max_x =  half_x - margin_x
    inner_min_y = -half_y + margin_y
    inner_max_y =  half_y - margin_y

    step_x = (inner_max_x - inner_min_x) / max(cols - 1, 1)
    step_y = (inner_max_y - inner_min_y) / max(rows - 1, 1)

    xs_local = [inner_min_x + i * step_x for i in range(cols)]
    ys_local = [inner_min_y + j * step_y for j in range(rows)]

    base_pos, base_orn = p.getBasePositionAndOrientation(case_id)

    def to_world(x_l: float, y_l: float):
        pos_w, _ = p.multiplyTransforms(base_pos, base_orn, [x_l, y_l, 0.0], [0, 0, 0, 1])
        return (pos_w[0], pos_w[1])

    # Order chosen to match original mapping used for objects_in_order
    bottom = ys_local[0]
    top = ys_local[-1]
    x0, x1, x2, x3 = xs_local[0], xs_local[1], xs_local[2], xs_local[3]

    result = [
        to_world(x0, top),     # 1
        to_world(x0, bottom),  # 2
        to_world(x3, bottom),  # 3
        to_world(x2, top),     # 4
        to_world(x1, bottom),  # 5
        to_world(x1, top),     # 6
        to_world(x2, bottom),  # 7
        to_world(x3, top),     # 8
    ]
    return result

# Build object list and target placements based on the provided figure
# Initial arrangement (1..8):
# 1: box_4, 2: box_5, 3: box_6, 4: box_2, 5: cylinder_0, 6: box_3, 7: triangle, 8: box_0
objects_in_order = [
    ("box_4", box_4),
    ("box_5", box_5),
    ("box_6", box_6),
    ("box_2", box_2),
    ("cylinder_0", cylinder_0),
    ("box_3", box_3),
    ("triangle", triangle),
    ("box_0", box_0),
]

# Target XY are computed adaptively from the current case pose and size
target_xy_list = compute_case_slot_world_xy(case)

# Heights and motion parameters
hover_z = 0.86           # high above table but a bit closer for faster approach
grasp_z_offset = 0.04    # legacy (kept for fallback); main logic uses AABB-based height
place_z_offset = 0.065   # slightly above case top when releasing
approach_time = 0.8

# Helper to move to a Cartesian pose with via-points
def move_to_pose(target_pos, orn, via_z_first: float = None):
    """Move to pose using via-points (up then down) to avoid collisions."""
    ee_pos, _ = get_ee_world_pose(robot, ee_link_index)
    start = ee_pos
    if via_z_first is not None:
        up1 = [start[0], start[1], via_z_first]
        up2 = [target_pos[0], target_pos[1], via_z_first]
        move_ee_linear(robot, arm_joint_indices, ee_link_index, start, up1, orn, duration=approach_time)
        move_ee_linear(robot, arm_joint_indices, ee_link_index, up1, up2, orn, duration=approach_time)
        move_ee_linear(robot, arm_joint_indices, ee_link_index, up2, target_pos, orn, duration=approach_time)
    else:
        move_ee_linear(robot, arm_joint_indices, ee_link_index, start, target_pos, orn, duration=approach_time)


def get_body_aabb(body_id: int):
    """Return (aabb_min, aabb_max) of a body in world coordinates."""
    return p.getAABB(body_id, -1)


def compute_pick_heights(body_id: int, name: str):
    """Compute pre-touch and grasp Z based on object's world AABB."""
    aabb_min, aabb_max = get_body_aabb(body_id)
    bottom_z = aabb_min[2]
    top_z = aabb_max[2]
    height = max(0.005, top_z - bottom_z)
    # Choose grasp depth ratio per shape for better stability
    ratio = 0.55
    if "cylinder" in name:
        ratio = 0.60
    if "triangle" in name:
        ratio = 0.50
    grasp_z = bottom_z + height * ratio
    # Safety clamps
    grasp_z = max(bottom_z + 0.012, min(grasp_z, top_z - 0.004))
    pre_touch_z = top_z + 0.008
    return pre_touch_z, grasp_z


# Execute pick-and-place for each object
for idx, (name, body_id) in enumerate(objects_in_order):
    # Get current object base pose
    obj_pos, obj_orn = p.getBasePositionAndOrientation(body_id)
    pick_xy = obj_pos[:2]
    pick_above = [pick_xy[0], pick_xy[1], hover_z]
    pre_z, grasp_z = compute_pick_heights(body_id, name)
    pick_pre = [pick_xy[0], pick_xy[1], pre_z]
    pick_down = [pick_xy[0], pick_xy[1], grasp_z]

    # Move above object -> descend -> grasp -> lift
    move_to_pose(pick_above, downward_orn, via_z_first=hover_z)

    # For triangle piece, slow down and use a slightly higher grasp to improve stability
    if name == "triangle":
        approach_time = 1.2
    else:
        approach_time = 0.8

    # Two-step descent for reliable alignment
    move_to_pose(pick_pre, downward_orn)
    move_to_pose(pick_down, downward_orn)
    set_gripper(robot, finger_joint_indices, width=0.0)  # close to grasp
    step_for_seconds(0.6)
    move_to_pose(pick_pre, downward_orn)
    step_for_seconds(0.1)
    move_to_pose(pick_above, downward_orn)
    step_for_seconds(0.2)

    # Determine place target
    tgt_xy = target_xy_list[idx]
    case_pos, case_orn = p.getBasePositionAndOrientation(case)
    place_above = [tgt_xy[0], tgt_xy[1], hover_z]
    place_down = [tgt_xy[0], tgt_xy[1], case_pos[2] + place_z_offset]

    # Move above case slot -> descend -> release -> retreat
    move_to_pose(place_above, downward_orn, via_z_first=hover_z)
    move_to_pose(place_down, downward_orn)
    step_for_seconds(0.1)
    # Slight shake before release helps seating
    jitter = 0.006
    for dx in [-jitter, jitter, 0.0]:
        pos_now, _ = get_ee_world_pose(robot, ee_link_index)
        move_ee_linear(robot, arm_joint_indices, ee_link_index, pos_now, [pos_now[0] + dx, pos_now[1], pos_now[2]], downward_orn, duration=0.2, steps=40)
    set_gripper(robot, finger_joint_indices, width=0.04)  # open
    step_for_seconds(0.4)
    move_to_pose(place_above, downward_orn)
    step_for_seconds(0.2)

# Return to home and idle visualization
go_home(robot, arm_joint_indices)
step_for_seconds(2.0)

p.disconnect()

