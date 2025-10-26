import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import time
import os
from threading import Thread
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
from ament_index_python.packages import get_package_share_directory


class AutoDriveNode(Node):
    def __init__(self):
        super().__init__('auto_drive_node')
        self.vel_pub = self.create_publisher(Twist, '/cmd_vel_unstamped', 1)

        self.timer = self.create_timer(0.02, self.timer_callback)

        self.bridge = CvBridge()
        self.img_sub = self.create_subscription(Image,
                                                '/oakd/rgb/preview/image_raw',
                                                self.img_callback,
                                                1)

        self.linear_speed = 1.2
        self.yaw_speed = 0.0


        # load pattern images from package share (robust to workspace location)
        try:
            pkg_share = get_package_share_directory('auto_drive')
        except Exception:
            # fallback: assume workspace root layout
            pkg_share = os.path.join(os.getcwd(), '..')

        self.red_pattern = cv2.imread(os.path.join(pkg_share, 'picture', 'red_pattern.png'))
        self.green_pattern = cv2.imread(os.path.join(pkg_share, 'picture', 'green_pattern.png'))
        self.stop_pattern = cv2.imread(os.path.join(pkg_share, 'picture', 'stop_pattern.png'))

        # navigation flag used in callback; ensure it's always defined
        self.start_navigation = False

        # verify pattern images loaded correctly
        self.patterns_loaded = True
        if self.red_pattern is None:
            self.get_logger().warning(f'red_pattern not found at {os.path.join(pkg_share, "picture", "red_pattern.png")}')
            self.patterns_loaded = False
        if self.green_pattern is None:
            self.get_logger().warning(f'green_pattern not found at {os.path.join(pkg_share, "picture", "green_pattern.png")}')
            self.patterns_loaded = False
        if self.stop_pattern is None:
            self.get_logger().warning(f'stop_pattern not found at {os.path.join(pkg_share, "picture", "stop_pattern.png")}')
            self.patterns_loaded = False

        # counter/flag to ensure stop-handling executes only once
        self.stop_count = 0

        self.cnt = 0

        self.create_green_light_cmd = """gz service --service /world/drive_track/create --reqtype gz.msgs.EntityFactory --reptype gz.msgs.Boolean --timeout 2000 --req 'name: "green_light"; sdf_filename: "/root/ros2_jazzy_ws/src/auto_drive_car/turtlebot4_simulator/turtlebot4_gz_bringup/worlds/green_light.sdf"; pose: {position: {x: -2, y: 4, z: 0}, orientation: {x: 0, y: 0, z: 1, w: 0}}'"""
        self.create_red_light_cmd = """gz service --service /world/drive_track/create --reqtype gz.msgs.EntityFactory --reptype gz.msgs.Boolean --timeout 2000 --req 'name: "red_light"; sdf_filename: "/root/ros2_jazzy_ws/src/auto_drive_car/turtlebot4_simulator/turtlebot4_gz_bringup/worlds/red_light.sdf"; pose: {position: {x: -2, y: 4, z: 0}, orientation: {x: 0, y: 0, z: 1, w: 0}}'"""
        self.remove_red_light_cmd = """gz service --service /world/drive_track/remove --reqtype gz.msgs.Entity --reptype gz.msgs.Boolean --timeout 2000 --req 'type: MODEL; name: "red_light"'"""
 
        os.system(self.create_red_light_cmd)

        self.change_traffic_light_thread = Thread(target=self.change_traffic_light)


    def change_traffic_light(self):
        time.sleep(7)
        os.system(self.create_green_light_cmd)
        os.system(self.remove_red_light_cmd)


    def img_callback(self, msg):
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error('bridge.imgmsg_to_cv2 failed')
            return

        hsv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)
        lower_yellow = np.array([15, 50, 50])
        upper_yellow = np.array([45, 255, 255])

        mask = cv2.inRange(hsv_img, lower_yellow, upper_yellow)

        cv2.imshow('img', cv_img)
        cv2.waitKey(1)

        mask = mask[220:, :]

        moments = cv2.moments(mask)

        if moments['m00'] == 0 or self.start_navigation == True:
            self.timer.destroy()
            return

        now_traffic_light_pattern = cv_img[5:133, 270:315, :]
        now_stop_sign_pattern = cv_img[20:128, 2:113, :]

        # if pattern images are available, compute MSE and act accordingly
        if self.patterns_loaded:
            try:
                red_diff = now_traffic_light_pattern.astype("float") - self.red_pattern.astype("float")
                squared_red_diff = np.square(red_diff)
                red_mse = np.mean(squared_red_diff)

                green_diff = now_traffic_light_pattern.astype("float") - self.green_pattern.astype("float")
                squared_green_diff = np.square(green_diff)
                green_mse = np.mean(squared_green_diff)

                stop_diff = now_stop_sign_pattern.astype("float") - self.stop_pattern.astype("float")
                squared_stop_diff = np.square(stop_diff)
                stop_mse = np.mean(squared_stop_diff)
                print(f'red_mse: {red_mse}; green_mse: {green_mse}; stop_mse: {stop_mse}')
                if red_mse < 1000 and red_mse < green_mse:
                    self.linear_speed = 0.0
                    if not self.change_traffic_light_thread.is_alive():
                        self.change_traffic_light_thread.start()
                if green_mse < 4000 and green_mse < red_mse:
                    self.linear_speed = 1.2
                # execute stop-handling only once
                if stop_mse < 20337 and self.stop_count == 0:
                    self.stop_count += 1
                    self.get_logger().info('Stop sign detected — executing one-time stop behavior')
                    self.linear_speed = 0
                    time.sleep(7)
                    self.linear_speed = 1.2
                    time.sleep(2)
            except Exception as e:
                self.get_logger().error(f'pattern comparison failed: {e}')
                # fall back: keep previous linear_speed
        else:
            # patterns not loaded: skip comparison, optionally log once
            pass

        try:
            x_center = moments['m10'] / moments['m00']
            self.yaw_speed = (160 - x_center) * 0.08
        except:
            self.get_logger().error("ERROR:moments['m00'] == 0")
            self.yaw_speed = 0.0
            self.linear_velocity=0.0
            pass


    def timer_callback(self):
        msg = Twist()
        msg.angular.z = self.yaw_speed
        msg.linear.x = self.linear_speed
        self.vel_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)

    ad_node = AutoDriveNode()

    rclpy.spin(ad_node)

    ad_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()