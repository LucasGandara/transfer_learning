import math

import rospy
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64
from tf.transformations import euler_from_quaternion


class DebugReward(object):
    def __init__(self):
        rospy.Subscriber("odom", Odometry, self.odom_callback, queue_size=10)
        self.orientation = -100
        self.goal_angle = -100
        self.heading = -100

        self.goal_y = 0.0
        self.goal_x = 1.5

        self.orientation_publisher = rospy.Publisher(
            "/orientation", Float64, queue_size=10
        )
        self.goal_angle_publisher = rospy.Publisher(
            "/goal_angle", Float64, queue_size=10
        )
        self.heading_publisher = rospy.Publisher("/heading", Float64, queue_size=10)

    def odom_callback(self, odom: Odometry):
        self.position = odom.pose.pose.position
        orientation = odom.pose.pose.orientation
        orientation_list = [orientation.x, orientation.y, orientation.z, orientation.w]
        _, _, yaw = euler_from_quaternion(orientation_list)

        self.orientation = yaw
        self.goal_angle = math.atan2(
            self.goal_y - self.position.y, self.goal_x - self.position.x
        )

        self.heading = self.goal_angle - yaw
        if self.heading > math.pi:
            self.heading -= 2 * math.pi

        elif self.heading < -math.pi:
            self.heading += 2 * math.pi

        self.heading = round(self.heading, 2)

    def pub_msgs(self):
        orientation_msg = Float64()
        orientation_msg.data = self.orientation
        goal_angle_msg = Float64()
        goal_angle_msg.data = self.goal_angle
        heading_msg = Float64()
        heading_msg.data = self.heading

        self.orientation_publisher.publish(orientation_msg)
        self.goal_angle_publisher.publish(goal_angle_msg)
        self.heading_publisher.publish(heading_msg)


if __name__ == "__main__":
    rospy.init_node("debug_reward", anonymous=True)
    rospy.loginfo("Debug reward node started")

    node = DebugReward()
    while True:
        node.pub_msgs()
        rospy.sleep(0.1)
