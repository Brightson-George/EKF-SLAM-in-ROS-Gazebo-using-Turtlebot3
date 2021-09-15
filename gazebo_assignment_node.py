#!/usr/bin/env python

import rospy  
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from math import atan2,sin,cos
from tf.transformations import euler_from_quaternion
from geometry_msgs.msg import Twist
import numpy as np
from IAMR_project.msg import rmse
from math import pi

#####[ Definition and Initializing ]#####

global x
global y
global theta
global x_d
global y_d
global i
global it

it=0
x_d=[1.0,-1.0,-1.0,-3.2,]
y_d=[5.0,1.0,-4.0,-0.9,]
i=0
x=-9.0;y=5.0;theta=0.0;
#x_d=10;y_d=10;
#angle_d=atan2(y_d-y,x_d-x)
Kp=0.2;Kd=0.2; Ki=0.0005
e_1=0
e_2=0
E=0
obst_range=list()
int_state=np.zeros((3,1))
true_x=-9.0
true_y=5.0
true_theta=0
true_states=np.array([[true_x],[true_y],[true_theta]])
global rmse1
rmse1=np.zeros((3,1))

####[ True states from odom callback ]####

def callback4(msg):
    global true_x
    global true_y
    global true_theta
    global true_states
    
    true_x=msg.pose.pose.position.x
    true_y=msg.pose.pose.position.y
    z=msg.pose.pose.position.z
    rot_q=msg.pose.pose.orientation
    (roll,pitch,true_theta)=euler_from_quaternion([rot_q.x,rot_q.y, rot_q.z,rot_q.w])
    true_states=np.array([[true_x],[true_y],[true_theta]])

####[ State Estimates from EKF Callback ]####

def Callback(msg):
	global x
	global y
	global theta
	global int_state

	x=msg.pose.pose.position.x
	y=msg.pose.pose.position.y
	theta=msg.pose.pose.orientation.z%(2*(pi))
	int_state=np.array([[x],[y],[theta]])

####[ Range of LaserScan sensor for Obstacles callback]####

def cb(obj):
	global obst_range
	
	obst_range=obj.ranges
	#rospy.loginfo(obst_range)

####[ Publishers and Subscribers ]####

rospy.init_node('waypointnav',anonymous=True)	#different name has no effect
#sub=rospy.Subscriber('odom',Odometry,Callback)
sub=rospy.Subscriber('state_estimate',Odometry,Callback)
sub3=rospy.Subscriber('/odom',Odometry,callback4)
sub_scan=rospy.Subscriber('/scan',LaserScan,cb)
pub1=rospy.Publisher('cmd_vel', Twist, queue_size=5)
pub_rmse=rospy.Publisher('rmse_topic',rmse,queue_size=5)	#RMSE value publisher topic
rmse_=rmse()	#custom msg
a=Twist()

r=rospy.Rate(10)
while not rospy.is_shutdown():
	
	it=it+1
	e_old=e_1
	angle_d=atan2(y_d[i]-y,x_d[i]-x)
	e=angle_d-theta
	e_1=atan2(sin(e),cos(e))
	E=E+e_1
	e_2=abs(x_d[i]-x)+abs(y_d[i]-y)
	linear_vel=0.05
	angular_vel=Kp*e_1 + Kd*(e_1-e_old) #+ Ki*E

####[ Obstacle avoidance behaviour ]####

	if all(f>1 for f in obst_range):
		a.linear.x=linear_vel
		a.linear.y=0.0
		a.angular.z=angular_vel
		pub1.publish(a)
	elif any(f<=1 for f in obst_range):	#Checks for range withing limits
		obst_rangetemp=[5 if x==float('inf') else x for x in obst_range]
		#obst_range[obst_range==float('inf')]=5
		if sum(obst_rangetemp[:180]) >= sum(obst_rangetemp[180:]):
		   a.linear.x=0.05
		   a.linear.y=0.0
		   a.angular.z=-0.3
		   pub1.publish(a)
		else:
		   a.linear.x=0.05
		   a.linear.y=0.0
		   a.angular.z=0.3
		   pub1.publish(a)
	#rospy.loginfo(a)
	
	if e_2<=0.1:	#Condition to terminate the navigation once a waypoint is reached
		a.linear.x=0.0
		a.angular.z=0.0001
		a.linear.y=0.0
		pub1.publish(a)
		E=0
		i+=1
		if i==4:
		    a.linear.x=0.0
		    a.angular.z=0.0001
		    a.linear.y=0.0
		    pub1.publish(a)
		rmse_.rmse_data=list(np.sqrt(rmse1/it))		#RMSE value is calculated at every waypoint and published
		pub_rmse.publish(rmse_)
		continue;
	rmse1=rmse1+(true_states-int_state)**2
	
	#rospy.loginfo(np.sqrt(rmse1/it))	
	r.sleep()
#if  __name__=='__main__':
# try:
# controller()
# except rospy.ROSInterruptException:
# pass
