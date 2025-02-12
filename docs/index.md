# Transfer learning

The *Transfer learning* proyect is My Masters degree Thesis. I'm trying to evaluate the improvement of
an agent which knowledge is transfered from a source model rather than training it from scratch.

## Start the poyect

### Prerequisites
- [git](https://git-scm.com/)
- [miniconda](https://docs.anaconda.com/miniconda/).
- [ROS Noetic - Full Desktop](http://wiki.ros.org/noetic/Installation/Ubuntu)

### QuickStart
Download this repo with git:
```sh
cd ~/catkin_ws/src/
git clone -b master https://github.com/LucasGandara/transfer_learning.git
cd ~/catkin_ws && catkin_make
```
Create a pyhon environment and install the dependencies using miniconda:
```sh
conda create -n tl python=3.9
conda activate tl
pip install -U rospkg
pip install -r requirements.txt
```

launch `main_track.launch` launch file:
```sh
roslaunch transfer_learning main_track.launch
```

## Potential Mobile bases
- [Husarion Rosbot](https://robots.ros.org/husarion-rosbot-2-pro/)

## Useful documentation
- [spawn_urdf package](https://github.com/ros-simulation/gazebo_ros_pkgs/blob/noetic-devel/gazebo_ros/scripts/spawn_model)
