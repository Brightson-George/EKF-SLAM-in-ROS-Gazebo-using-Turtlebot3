#!/usr/bin/env python
import numpy as np
import csv

import matplotlib.pyplot as plt
from sympy import sin,cos,Matrix,eye,atan2
from sympy.abc import x,y,v,w,theta,T,m,n
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion
import math 
import roslib
import tf
import numpy as np
from IAMR_project.msg import landmark_msgs, rmse
import geometry_msgs.msg
import turtlesim.srv
import apriltag_ros.msg
from apriltag_ros.msg import AprilTagDetectionArray
from apriltag_ros.msg import AprilTagDetection
import rospy  
from geometry_msgs.msg import Twist
from scipy import linalg

#####[ Initializing ]#####

true_x=-9.0
true_y=5.0
true_theta=0
tem =np.full(32, np.nan)
t_ =0.0001
v_data=0.01
w_data =0.01
true_states=np.array([[true_x],[true_y],[true_theta]])
global rmse1
rmse1=np.zeros((3,1))

####[ landmark data from camera callback ]####

def callback2(msg):
    global tem
    global t_
    tem=msg.landmark_data
    t_=msg.header.stamp.nsecs
    t_=float(t_/(10**9))

####[ linear and angular velocity from cmd_vel callback ]####

def callback3(msg):
    global v_data
    global w_data
    v_x=msg.linear.x
    v_y=0.0
    v_data=(v_x**2 + v_y**2)**0.5 + 0.001 * np.random.randn() + 0.0
    w_data=msg.angular.z + 0.004 * np.random.randn() + 0.0

####[ True states for odom callback ]####

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

##############################################################################[ EKF_Localization ]##################################################################3

def EKF(X,P,u,z,mj):
    F=np.eye(3);
    Q=sigma*np.eye(2);
    val={T:T1,v:u[0],w:u[1],theta:float(X[2])}
    G=J_mat.subs(val)
    G=np.array(G).astype(np.float64)
    #rospy.loginfo(G)
    V=I_mat.subs(val)
    V=np.array(V).astype(np.float64)
    X=np.dot(F,X)+U
    P=((G).dot(P).dot(G.T))+((V).dot(Mot).dot(V.T))
    for j in range(N_feat):
        zj=np.array([[z[2*j]],[z[2*j+1]]])
        if zj[0]*0==0:
            z_meas=np.array([[float(((mj[j,0]-X[0])**2+(mj[j,1]-X[1])**2)**0.5)],
                            [float(math.atan2((mj[j,1]-X[1]),(mj[j,0]-X[0]))-X[2])]])
            
            val1={m:zj[0],n:zj[1],x:X[0],y:X[1],theta:X[2]}
            H=K_mat.subs(val1)
            H=np.array(H).astype(np.float64)
            
            temp=(H).dot(P).dot(H.T)+Q
            K=(P).dot(H.T).dot(np.linalg.inv(temp))
            temp2=zj-z_meas
            
            X=np.matmul(K,temp2)+X
            P=np.dot((np.eye(3)-np.dot(K,H)),P)
    return X,P

###################################################################[ EKF_SLAM_known_Correspondence ]################################################################################

def EKF_SLAM_known_C(X,P,u,z,c):
    N=N2
    F=np.eye(3)
    F=np.concatenate((F, np.zeros((3,3*N))), axis=1)
    X_=X+np.dot(F.T,U)
    val={T:T1,v:u[0],w:u[1],theta:float(X[2])}
    G=J_mat.subs(val)
    G=np.array(G).astype(np.float64)
    G=G-np.eye(3)
    G=np.eye(3*N+3)+((F.T).dot(G).dot(F))
    P=((G).dot(P).dot(G.T))+((F.T).dot(R).dot(F))
    Q_t=np.array([[2,0,0],[0,2,0],[0,0,2]])
    
    for a in range(N_feat):
        zj=np.array([[z[2*a]],[z[2*a+1]],[a+1]])
        if (zj[0]*0==0) & (zj[1]*0==0):
            j=a+1
            if X[a*3+5]==0:
#                X[a*3+3,a*3+4,a*3+5]=np.array([X[0]+zj[0]*math.cos(z[1]+X[2]),X[1]+zj[0]*math.sin(z[1]+X[2]),0])
                X_[a*3+3]=X_[0]+zj[0]*math.cos(zj[1]+X_[2])
                X_[a*3+4]=X_[1]+zj[0]*math.sin(zj[1]+X_[2])
                X_[a*3+5]=j+0
                
                
            delta=np.array([[float(X_[a*3+3]-X_[0])],
                            [float(X_[a*3+4]-X_[1])]])
            q_=np.dot(delta.T,delta)#.astype(np.float64)
            q_=float(q_[0])
            q_sq=float(q_**0.5)
            z_hat=np.array([[q_sq],[float(math.atan2(delta[1],delta[0])-X_[2])],X_[a*3+5]])
            F_xj=np.concatenate((np.zeros((3,3+(3*j)-3)),np.eye(3),np.zeros((3,3*N-3*j))), axis=1)
            F_xj=np.concatenate((F,F_xj))
#            q_sq=np.linalg.sqrtm(q)
            dx=float(delta[0])
            dy=float(delta[1])
            H_j=np.array([[-q_sq*dx, -q_sq*dy , 0 , +q_sq*dx , +q_sq*dy , 0 ],
                          [   dy   ,     -dx  ,-q_ ,    -dy  ,   +dx    , 0 ],
                          [    0   ,     0    , 0 ,     0   ,    0     , q_]])
            H_j=(1/q_)*H_j.dot(F_xj)
            temp=(H_j).dot(P).dot(H_j.T) + Q_t 
            temp1=np.linalg.inv(temp)
            K_j= (P).dot(H_j.T).dot(temp1)#.astype(np.float64)
            X_=X_ + ((K_j).dot(zj-z_hat))#.astype(np.float64)
            P_=(np.eye(3*N+3)-((K_j).dot(H_j)))
            P=np.dot(P_,P)#.astype(np.float64)
    X=X_
#    P=P
    return X_,P

################################################################################[ EKF_SLAM_Unknown_Correspondence ]##########################################################


def EKF_SLAM_unknown_C(X,P,u,z,N):
    Nt=N
    F=np.eye(3)
    F=np.concatenate((F, np.zeros((3,3*Nt))), axis=1)
    X_=X+np.dot(F.T,U)
    val={T:T1,v:u[0],w:u[1],theta:float(X[2])}
    G=J_mat.subs(val)
    G=np.array(G).astype(np.float64)
    G=G-np.eye(3)
    G=np.eye(3*N+3)+((F.T).dot(G).dot(F))
    P=((G).dot(P).dot(G.T))+((F.T).dot(R).dot(F))
    Q_t=np.array([[2,0,0],[0,2,0],[0,0,2]])
    
    for a in range(N_feat):
        zj=np.array([[z[2*a]],[z[2*a+1]],[a+1]])
        pi_k=np.array([])
        if (zj[0]*0==0) & (zj[1]*0==0):
            j=a+1
            
#                X[a*3+3,a*3+4,a*3+5]=np.array([X[0]+zj[0]*math.cos(z[1]+X[2]),X[1]+zj[0]*math.sin(z[1]+X[2]),0])
            LM=np.array([X_[0]+zj[0]*float(math.cos(zj[1]+X_[2])),
                       X_[1]+zj[0]*float(math.sin(zj[1]+X_[2])),
                       [j]])
  
                
            for i in range(Nt):   
                delta=np.array([[float(X_[i*3+3]-X_[0])],
                                [float(X_[i*3+4]-X_[1])]])
                q_=np.dot(delta.T,delta)#.astype(np.float64)
                q_=float(q_[0])
                q_sq=float(q_**0.5)
                
                z_hat=np.array([[q_sq],[float(math.atan2(delta[1],delta[0])-X_[2])],X_[i*3+5]])
                F_xj=np.concatenate((np.zeros((3,3+(3*(i+1)-3))),np.eye(3),np.zeros((3,3*(Nt)-3*(i+1)))),axis=1)
                F1=np.eye(3)
                F1=np.concatenate((F1, np.zeros((3,3*Nt))), axis=1)
                F_xj=np.concatenate((F1,F_xj))
    #            q_sq=np.linalg.sqrtm(q)
                dx=float(delta[0])
                dy=float(delta[1])
                H_j=np.array([[-q_sq*dx, -q_sq*dy , 0 , +q_sq*dx , +q_sq*dy , 0 ],
                              [   dy   ,     -dx  ,-q_ ,    -dy  ,   +dx    , 0 ],
                              [    0   ,     0    , 0 ,     0   ,    0     , q_]])
                H_j=(1/q_)*(H_j).dot(F_xj)
                psi=(H_j).dot(P).dot(H_j.T) + Q_t 
                psi_inv=np.linalg.inv(psi)
                pik=((zj-z_hat).T).dot(psi_inv).dot(zj-z_hat)
                pi_k=np.append(pi_k,pik)
                
            pi_k=np.append(pi_k,1)
            j_i=np.argmin(pi_k)
            N=Nt
            Nt=np.maximum(Nt,j_i+1)
            
            if Nt !=N:
                X_=np.append(X_,LM,axis=0)
                P=linalg.block_diag(P,np.eye(3))
        
            delta=np.array([[float(X_[(j_i)*3+3]-X_[0])],
                                [float(X_[(j_i)*3+4]-X_[1])]])
            q_=np.dot(delta.T,delta)#.astype(np.float64)
            q_=float(q_[0])
            q_sq=float(q_**0.5)
            z_hat=np.array([[q_sq],[float(math.atan2(delta[1],delta[0])-X_[2])],X_[(j_i)*3+5]])
            F_xj=np.concatenate((np.zeros((3,3+(3*(j_i+1))-3)),np.eye(3),np.zeros((3,3*Nt-3*(j_i+1)))), axis=1)
            F2=np.eye(3)
            F2=np.concatenate((F2, np.zeros((3,3*Nt))), axis=1)
            F_xj=np.concatenate((F2,F_xj))
    #       q_sq=np.linalg.sqrtm(q)
            dx=float(delta[0])
            dy=float(delta[1])
            H_j=np.array([[-q_sq*dx, -q_sq*dy , 0 , +q_sq*dx , +q_sq*dy , 0 ],
                          [   dy   ,     -dx  ,-q_ ,    -dy  ,   +dx    , 0 ],
                          [    0   ,     0    , 0 ,     0   ,    0     , q_]])
            H_j=(1/q_)*(H_j).dot(F_xj)
            psi=(H_j).dot(P).dot(H_j.T) + Q_t
            psi_inv=np.linalg.inv(psi)
            K_j= (P).dot(H_j.T).dot(psi_inv)#.astype(np.float64)
            X_=X_ + ((K_j).dot(zj-z_hat))#.astype(np.float64)
            P=(np.eye(3*Nt+3)-((K_j).dot(H_j))).dot(P)#.astype(np.float64)
            
    X=X_
#    P=P
    return X_,P,Nt
            
#################################################################################[ Main Function ]###############################################################################################

if __name__ == '__main__':

    rospy.init_node('EKF_localization',anonymous=True)

####  Finding the Jacobian of g-Matrix   #####
    #rospy.sleep(0.2)
    F_j=eye(3)
    x_j=Matrix([[x],[y],[theta]])
    U_j=Matrix([[-v*sin(theta)/w+v*sin(theta+w*T)/w],[v*cos(theta)/w-v*cos(theta+w*T)/w],[w*T]])
    h=Matrix([[((m-x)**2+(n-y)**2)**(1/2)],
           [atan2((n-y),(m-x))-theta]])
    

    G_fn=F_j*x_j + U_j
    D=Matrix([[x,y,theta]])
    E=Matrix([[v,w]])
    I_mat=G_fn.jacobian(E)
    J_mat=G_fn.jacobian(D)
    K_mat=h.jacobian(D)

#################################[ Subscribers and Publishers ]##########################################

    sub=rospy.Subscriber('/landmark_topic',landmark_msgs,callback2)
    sub2=rospy.Subscriber('/cmd_vel',Twist,callback3)
    sub3=rospy.Subscriber('/odom',Odometry,callback4)
    pub_est=rospy.Publisher('state_estimate',Odometry,queue_size=5)
    pub_land=rospy.Publisher('unknown_landmarks',rmse,queue_size=5) #to store the POS of unknown landmarks
    loc_=rmse()
    est=Odometry()

################################################################################

#Defining matrices and initial state variables 1.EKF localization	2.EKF Known Slam	3.EKF Unknown Slam 

    #1.EKF localization
    landmarkset_1=[14,15,16,17,20,21,22,23,24,25,28,29]
    global X
    state_est=np.array([[-9.0],[5.0],[0.0]]);P1=np.eye(3);
    X=state_est
    mj=np.array([[-1.0,7.5],[6.0,-6.0],[6.0,3.0],[-6.0,5.0],[-2.0,5.0],[-7.2,-3.0]])
    i=0
    rate = rospy.Rate(10)

    
    #2.EKF Known Slam
    landmarkset_2=[0,1,2,3,6,7,8,9,26,27]
    global X_slam2
    N2=5
    X_slam2=np.zeros((3+3*N2,1))
    X_slam2[0]=-9.0 ; X_slam2[1]=5.0
    P2=0.1*np.eye(3*N2+3)
    slam_known_c=X_slam2

    #3.EKF Unknown Slam 
    landmarkset_3=[4,5,10,11,12,13,18,19,30,31]
    global X_slam3
    Nt=0
    X_slam3=np.zeros((3+3*Nt,1))
    X_slam3[0]=-9.0
    X_slam3[1]=5.0
    P3=0.1*np.eye(3*Nt+3)
    slam_unk_c=X_slam3

#Common Parameters
    R=0.01*np.eye(3)
    sigma=1.5; alpha1=0.05;alpha2=0.3; alpha3=0.3; alpha4=0.05;

    while not rospy.is_shutdown():
        l1=np.array(tem)
	
	T1=0.1
        
    	u=np.array([[v_data],[w_data]])
    	U=np.array([[-v_data*math.sin(X[2])/float(w_data)+v_data*math.sin(X[2]+w_data*T1)/float(w_data)],
               [v_data*math.cos(X[2])/float(w_data)-v_data*math.cos(X[2]+w_data*T1)/float(w_data)],
               [w_data*T1]])
    	Mot=np.array([[(alpha1*v_data+alpha2*w_data)**2,0],[0,(alpha3*v_data+alpha4*w_data)**2]])

#################[ conditons to choose which algorithm ]##########################

	z1=l1[landmarkset_1] ; z2=l1[landmarkset_2] ; z3=l1[landmarkset_3]
	wt_z1=np.count_nonzero(~np.isnan(np.array(z1)))
	wt_z2=np.count_nonzero(~np.isnan(np.array(z2)))
	wt_z3=np.count_nonzero(~np.isnan(np.array(z3)))
	wts_=[4,wt_z2,wt_z3] ; wt_id=wts_.index(max(wts_))	#Higher weight assigned to EKF localization
	if wt_id==0:
	    X=state_est
	    N_feat=int(z1.shape[0]/2)
    	    loc_c,loc_b=EKF(X,P1,U,z1,mj)
	    state_est=loc_c
	    P1=loc_b
	elif wt_id==1:
  	    slam_known_c[:3]=state_est
	    X_slam2=slam_known_c
	    N_feat=int(z2.shape[0]/2)
	    c=np.array([1,2,3,4,5])
	    slam_known_c,slam_known_b=EKF_SLAM_known_C(X_slam2,P2,u,z2,c)
	    state_est=slam_known_c[:3]
	    P2=slam_known_b
	else:
	    slam_unk_c[:3]=state_est
	    X_slam3=slam_unk_c
	    N_feat=int(z3.shape[0]/2)
	    slam_unk_c,slam_unk_b,Nt=EKF_SLAM_unknown_C(X_slam3,P3,u,z3,Nt)
	    state_est=slam_unk_c[:3]
	    P3=slam_unk_b
		     


	#rospy.loginfo(state_est)

########[ Publishing the state estimates ]###########

	est.pose.pose.position.x=state_est[0]
    	est.pose.pose.position.y=state_est[1]
    	est.pose.pose.orientation.z=state_est[2]
    	pub_est.publish(est)

	i=i+1
	
########[ Publishing the Estimates of Landmarks position ]############
        loc_.unknown_landmark=list(X_slam2) 	#Give X_slam3 in place of X_slam2 to get the other landmarks positions
        pub_land.publish(loc_)
        rate.sleep()
        

