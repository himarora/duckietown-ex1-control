#!/usr/bin/env python3
import numpy as np
import rospy
from duckietown.dtros import DTROS, NodeType, TopicType, DTParam, ParamType
from duckietown_msgs.msg import Twist2DStamped, LanePose, SegmentList
from numpy.linalg import norm

from lane_controller.controller import PurePursuitLaneController
np.set_printoptions(precision=2, suppress=True)

class LaneControllerNode(DTROS):
    """Computes control action.
    The node compute the commands in form of linear and angular velocitie.
    The configuration parameters can be changed dynamically while the node is running via ``rosparam set`` commands.
    Args:
        node_name (:obj:`str`): a unique, descriptive name for the node that ROS will use
    Configuration:

    Publisher:
        ~car_cmd (:obj:`Twist2DStamped`): The computed control action
    Subscribers:
        ~lane_pose (:obj:`LanePose`): The lane pose estimate from the lane filter
    """

    def __init__(self, node_name):
        # Initialize the DTROS parent class
        super(LaneControllerNode, self).__init__(
            node_name=node_name,
            node_type=NodeType.CONTROL
        )

        # Add the node parameters to the parameters dictionary
        self.p = dict()
        self.pp_controller = PurePursuitLaneController(self.p)

        # Construct publishers
        self.pub_car_cmd = rospy.Publisher("~car_cmd",
                                           Twist2DStamped,
                                           queue_size=1,
                                           dt_topic_type=TopicType.CONTROL)

        # Construct subscribers
        self.sub_lane_reading = rospy.Subscriber("~lane_pose",
                                                 LanePose,
                                                 self.cbLanePoses,
                                                 queue_size=1)
        self.load_params()
        topic = "/agent/lane_filter_node/seglist_filtered"
        # topic = "/agent/ground_projection_node/lineseglist_out"
        self.sub0 = rospy.Subscriber(topic,
                                     SegmentList,
                                     self.cbSegmentsGround,
                                     queue_size=1)
        self.pose_msg = None
        self.mode = "straight"
        self.r_y = 0
        self.state = [np.array([0, 0])] * 5
        self.tj = np.array([0] * 2)
        self.log("Node Initialized!")
        self.log(f"Subscribed to {topic}")

    @staticmethod
    def filter_segments_array(segments_array, distance_threshold):
        # adjust color
        segments_array[:, 0][segments_array[:, 0] == 2.] = 1.

        # filter length
        segments_array = segments_array[segments_array[:, -1] <= distance_threshold]

        # sort by distance
        segments_array = segments_array[np.argsort(segments_array[:, -1])]
        return segments_array

    @staticmethod
    def parse_segment_list(segments_msg, dist_threshold):
        segments = segments_msg.segments
        data = []
        for s in segments:
            p, col = s.points, s.color
            p1, p2 = (x1, y1), (x2, y2) = (p[0].x, p[0].y), (p[1].x, p[1].y)
            c = np.mean((x1, x2)), np.mean((y1, y2))
            data.append((col, *p1, norm(p1), *p2, norm(p2), *c, norm(c)))
        data = np.array(data)
        data = LaneControllerNode.filter_segments_array(data, dist_threshold)
        return data

    @staticmethod
    def get_equation(segments_array, close_segs, far_segs):
        closest, farthest = segments_array[:close_segs].mean(axis=0), segments_array[-far_segs:].mean(axis=0)
        x1, y1 = closest[-3], closest[-2]
        x2, y2 = farthest[-3], farthest[-2]
        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1
        w_eq = np.array((m, b))
        f_pt = np.array((x2, y2))
        return w_eq, f_pt

    @staticmethod
    def get_trajectory(yl, wl, off):
        yellow_line_detected = True if yl is not None else False
        white_line_detected = True if wl is not None else False
        if yellow_line_detected or white_line_detected:
            if not white_line_detected:
                wl = np.array((yl[0] if wl is None else wl[0], yl[1] - 2 * off))
            elif not yellow_line_detected:
                yl = np.array((wl[0] if yl is None else yl[0], wl[1] + 2 * off))
            elif wl[1] is None:
                wl[1] = yl[1]
            elif yl[1] is None:
                yl[1] = wl[1]
        # wl = np.array(
        #     (yl[0] if wl is None or wl[1] is None else wl[0], yl[1] - 2 * off)) if not white_line_detected else wl
        # yl = np.array(
        #     (wl[0] if yl is None or yl[1] is None else yl[0], wl[1] + 2 * off)) if not yellow_line_detected else yl
        # assert yl is not None and wl is not None
        return (yl + wl) / 2.

    @staticmethod
    def handle_noise_straight(yl, y_segs, wl, w_segs, off):
        yellow_line_detected = True if yl is not None else False
        white_line_detected = True if wl is not None else False
        right_white = False
        grass_yellow = False

        if white_line_detected:
            # While going straight, if white line is detected on the left side and yellow line is not detected,
            # shift the yellow line two units, and white line 4 units away
            if w_segs[:, -2].mean() > 0:
                # print("white detected on left")
                # yl = np.array((wl[0], wl[1] - 2 * off)) if not yellow_line_detected else yl
                wl = None
                right_white = True

            # If both lines are detected but yellow line is detected on the right side of white line, assume that
            # it is grass and compute trajectory using white line
            elif yellow_line_detected and y_segs[:, -2].mean() < w_segs[:, -2].mean():
                yl = None
                grass_yellow = True
        return yl, wl, grass_yellow, right_white

    @staticmethod
    def get_lookahead_point(tj_eq, l, y_f, n_y, w_f, n_w):
        r_x = r_x1, r_x2 = np.roots(
            [2 * tj_eq[0] ** 2, 2 * tj_eq[0] * tj_eq[1], tj_eq[1] ** 2 - l ** 2])
        r_y1, r_y2 = tj_eq[0] * r_x + tj_eq[1]
        r1, r2 = np.array([r_x1, r_y1]), np.array([r_x2, r_y2])
        r = None
        if y_f is not None and w_f is not None:
            dir_pt = y_f if n_y > n_w else w_f
        elif y_f is not None and w_f is None:
            dir_pt = y_f
        elif w_f is not None and y_f is None:
            dir_pt = w_f
        if not r:
            r = r1 if norm(dir_pt - r1) < norm(dir_pt - r2) else r2
        return r

    @staticmethod
    def get_distance_from_line(point, line):
        a, b, c = -line[0], 1, -line[1]
        x, y = point
        dist = (np.abs(a * x + b * y + c)) / (np.sqrt(a ** 2 + b ** 2))
        return dist

    def load_params(self):
        def _init_dtparam(name):
            str_topics = []
            param_type = ParamType.STRING if name in str_topics else ParamType.FLOAT
            return DTParam(
                f'~{name}',
                param_type=param_type,
                min_value=-100.0,
                max_value=100.0
            )

        param_names = ["L", "th_seg_dist", "th_seg_count", "off_lane", "th_seg_close", "th_seg_far", "wt_slope",
                       "wt_dist", "th_turn_slope", "th_st_slope", "th_lane_slope", "pow_slope", "pow_dist", "v",
                       "wt_omega", "th_omega", "min_slope", "max_L", "exp"]
        self.p = {k: _init_dtparam(k) for k in param_names}

    def cbSegmentsGround(self, line_segments_msg):
        self.segments_msg = line_segments_msg
        self.load_params()
        data = self.parse_segment_list(line_segments_msg, self.p["th_seg_dist"].value)  # was 0.3
        white_segments, yellow_segments = data[data[:, 0] == 0.], data[data[:, 0] == 1.]
        n_white_segs, n_yellow_segs = len(white_segments), len(yellow_segments)
        line_detection_threshold = self.p["th_seg_count"].value
        white_detected, yellow_detected = n_white_segs > line_detection_threshold, n_yellow_segs > line_detection_threshold
        # If none of the lines are detected, assume that the trajectory has not changed
        gr_f, wr_f, wts_f = False, False, False
        if not white_detected and not yellow_detected:
            w_eq, y_eq, tj_eq, w_f, y_f = self.state

        # If any of the lines is detected, compute the new trajectory
        else:
            w_eq, w_f = LaneControllerNode.get_equation(white_segments, self.p["th_seg_close"].value,
                                                        self.p["th_seg_far"].value) if white_detected else (None, None)
            y_eq, y_f = LaneControllerNode.get_equation(yellow_segments, self.p["th_seg_close"].value,
                                                        self.p["th_seg_far"].value) if yellow_detected else (None, None)

            # If taking left turn and only yellow is detected
            prev_m = self.state[2][0]
            if y_eq is not None and w_eq is None and np.abs(prev_m) > self.p["th_turn_slope"].value and prev_m >= 0:
                y_eq = None
            elif w_eq is not None and y_eq is None and np.abs(prev_m) > self.p["th_turn_slope"].value and prev_m < 0:
                w_eq = None

            if w_eq is None and y_eq is None:
                w_eq, y_eq, tj_eq, w_f, y_f = self.state
            else:
                tj_slope = ((w_eq[0] if w_eq is not None else y_eq[0]) + (y_eq[0] if y_eq is not None else w_eq[0])) / 2.

                wts_f = False
                # Handle any noise
                if np.abs(tj_slope) <= self.p["th_lane_slope"].value:
                    y_eq, w_eq, gr_f, wr_f = self.handle_noise_straight(y_eq, yellow_segments, w_eq, white_segments,
                                                            self.p["off_lane"].value)
                # While turning right, only use the yellow line
                elif tj_slope < 0 and np.abs(tj_slope) > self.p[
                    "th_lane_slope"].value and yellow_detected and w_eq is not None:
                    w_eq[1] = None
                    wts_f = True

                tj_eq = self.get_trajectory(y_eq, w_eq, self.p["off_lane"].value)

        # y_eq =
        r = self.get_lookahead_point(tj_eq, self.p["L"].value, y_f, n_yellow_segs, w_f, n_white_segs)
        self.state = w_eq, y_eq, tj_eq, w_f, y_f,
        self.tj = tj_eq
        self.r_y = r[1]
        w_eq = np.array([0, 0]) if w_eq is None else w_eq
        y_eq = np.array([0, 0]) if y_eq is None else y_eq
        print(f"Y:{yellow_detected}{y_eq} W:{white_detected}{w_eq} Gr:{gr_f} Wr:{wr_f} Wts_f:{wts_f}")
        # print(y_eq[0] if y_eq is not None else "None", w_eq[0] if w_eq is not None else "None")
        # if w_eq is not None and y_eq is not None:
        #     print(y_eq, tj_eq, w_eq)
        #     print(self.get_distance_from_line(r, y_eq), self.get_distance_from_line(r, tj_eq), self.get_distance_from_line(r, w_eq))

    def cbLanePoses(self, input_pose_msg):
        """Callback receiving pose messages

        Args:
            input_pose_msg (:obj:`LanePose`): Message containing information about the current lane pose.
        """
        self.pose_msg = input_pose_msg
        # print()
        self.load_params()
        car_control_msg = Twist2DStamped()
        car_control_msg.header = self.pose_msg.header

        v = self.p["v"].value
        L = self.p["L"].value
        slope = np.abs(self.tj[0])
        # L *= (1 / np.exp(self.p["exp"].value * slope))
        if slope >= self.p["th_turn_slope"].value:
            L = 0.35
            v = 0.2
            # L_factor = 1 / (slope * self.p["wt_slope"].value)
            # L *= L_factor
        # L = min(L, self.p["max_L"].value)
        # print(L)
        # print(self.r_y)
        omega = (2 * v * self.r_y) / L ** 2  # Pure Pursuit
        # print(omega)
        # omega = 0
        # print(slope-1)
        # print(omega)seglist_filtered
        # print(omega)
        # Adjustments for turning
        # if np.abs(self.tj[0]) >= self.p[
        #     "th_turn_slope"].value:
        #     omega *= np.abs(
        #         self.p["wt_slope"].value * self.tj[0] ** self.p["pow_slope"].value * self.p["wt_dist"].value * (
        #                 self.tj[1] / self.tj[0]) ** self.p["pow_dist"].value)
        # else:
        #     omega *= 1

        # if np.abs(omega) > self.p["th_omega"].value:
        #     v *= np.abs(omega) * self.p["wt_omega"].value

        # v *= 1 / ((np.abs(omega) + 1) * self.p["wt_omega"].value)
        # print(np.abs(omega) + 1, v)
        car_control_msg.v = v
        car_control_msg.omega = omega
        # self.log(car_control_msg.omega)
        self.publishCmd(car_control_msg)

    def publishCmd(self, car_cmd_msg):
        """Publishes a car command message.

        Args:
            car_cmd_msg (:obj:`Twist2DStamped`): Message containing the requested control action.
        """
        self.pub_car_cmd.publish(car_cmd_msg)

    def cbParametersChanged(self):
        """Updates parameters in the controller object."""

        self.pp_controller.update_parameters(self.p)


if __name__ == "__main__":
    # Initialize the node
    lane_controller_node = LaneControllerNode(node_name='lane_controller_node')
    # Keep it spinning
    rospy.spin()

######
# Input pose message:
# header:
#   seq: 1
#   stamp:
#     secs: 1602516449
#     nsecs: 539887666
#   frame_id: ''
# d: 0.0
# d_ref: 0.0
# sigma_d: 0.0
# phi: 0.05000000074505806
# phi_ref: 0.0
# sigma_phi: 0.0
# curvature: 0.0
# curvature_ref: 0.0
# v_ref: 0.0
# status: 0
# in_lane: True

# Twist2DStamped car_control_msg
# header:
#   seq: 0
#   stamp:
#     secs: 0
#     nsecs:         0
#   frame_id: ''
# v: 0.0
# omega: 0.0


# car_control_msg with pose_msg header
# header:
#   seq: 64
#   stamp:
#     secs: 1602516455
#     nsecs: 435892581
#   frame_id: ''
# v: 0.0
# omega: 0.0


# Final car_control_msg
# header:
#   seq: 66
#   stamp:
#     secs: 1602516455
#     nsecs: 627885341
#   frame_id: ''
# v: 0.5
# omega: 0


# [INFO] [1602618756.701883]: [/agent/lane_controller_node] cbLineSegments:
# header:
#   seq: 55
#   stamp:
#     secs: 1602618756
#     nsecs: 686120271
#   frame_id: ''
# segments:
#   -
#     color: 0
#     pixels_normalized:
#       -
#         x: 0.0
#         y: 0.0
#       -
#         x: 0.0
#         y: 0.0
#     normal:
#       x: 0.0
#       y: 0.0
#     points:
#       -
#         x: 0.6783292764044644
#         y: 0.41251615715755474
#         z: 0.0
#       -
#         x: 0.657085132790152
#         y: 0.3657558537380689
#         z: 0.0
#   -
#     color: 0
#     pixels_normalized:
#       -
#         x: 0.0
#         y: 0.0
#       -
#         x: 0.0
#         y: 0.0
#     normal:
#       x: 0.0
#       y: 0.0
#     points:
#       -
#         x: 1.4033407531117965
#         y: 2.002345373199374
#         z: 0.0
#       -
#         x: 2.6523429040678943
#         y: 3.463015048622708
#         z: 0.0
