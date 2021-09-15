#!/usr/bin/env python  
from nav_msgs.msg import Odometry
import roslib
#roslib.load_manifest('learning_tf')
import rospy  
import math
import tf
import numpy as np
from IAMR_project.msg import landmark_msgs
import geometry_msgs.msg
import turtlesim.srv
import apriltag_ros.msg
from apriltag_ros.msg import AprilTagDetectionArray
from apriltag_ros.msg import AprilTagDetection
from geometry_msgs.msg import Twist
global name
name={0: 'A', 1: 'B', 2: 'C', 3:'D', 4: 'E',5: 'F',6: 'G',7: 'H',8: 'I',9: 'J',10: 'K',11: 'L',12: 'M',13: 'N',14: 'O',15: 'P'}	#initializing the tag IDs with names to be used for lookup transform
id_list=[]

####[ Tag Detections from apriltag_ros detections callback ]####

def callback(msg):
    global id_arraysize
    global id_list
    #id_list=msg.detections[0].id
    id_tags=msg.detections
    for k in range(len(id_tags)):
	id_tuple=msg.detections[k].id
    	ids=id_tuple[0]
    	id_list.insert(k,ids)
    id_arraysize=len(id_list)


if __name__ == '__main__':
    rospy.init_node('tf_listener_node', anonymous=True)

    listener = tf.TransformListener()
    sub=rospy.Subscriber('/tag_detections',AprilTagDetectionArray,callback)	#subscribing to tag_detections
    pub_=rospy.Publisher('/landmark_topic',landmark_msgs,queue_size=1)
    msg_=landmark_msgs()

    rate = rospy.Rate(10.0)
    while not rospy.is_shutdown():
	
	data_from_camera=np.full(32, np.nan)
	data_from_camera=list(data_from_camera)
	for i in id_list:
           try:
                (trans,rot) = listener.lookupTransform('base_footprint', name[i], rospy.Time(0))	#lookup transform for every frame given by name[i] and base_footprint for all detected tags
		angular = math.atan2(trans[1], trans[0])						#calculate range and bearing
		linear = math.sqrt(trans[0] ** 2 + trans[1] ** 2)
		data_from_camera[2*i]=linear
		data_from_camera[2*i+1]=angular
		
		
		
           except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                continue

		#angular = 4 * math.atan2(trans[1], trans[0])
		#linear = 0.5 * math.sqrt(trans[0] ** 2 + trans[1] ** 2)
		#rospy.loginfo(linear)
		#rospy.loginfo(angular)
		#cmd = geometry_msgs.msg.Twist()
		#cmd.linear.x = linear
		#cmd.angular.z = angular
		#turtle_vel.publish(cmd)
		
	msg_.landmark_data=data_from_camera
	msg_.header.stamp=rospy.get_rostime()
	pub_.publish(msg_)
    rate.sleep()




