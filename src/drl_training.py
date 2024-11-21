#!/usr/bin/env python
# Authors: Lucas G. #

import rospy


class DRLTraining(object):
    def __init__(self):
        pass


if __name__ == "__main__":
    rospy.init_node("drl_training", log_level=rospy.INFO, anonymous=True)

    drl_training = DRLTraining()
