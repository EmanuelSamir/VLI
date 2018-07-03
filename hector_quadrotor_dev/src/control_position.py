#!/usr/bin/env python

import rospy
import numpy as np
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Twist
import time


class Drone(object):
	def __init__(self):
		self.cmd = rospy.Publisher("cmd_vel",Twist,queue_size=10)
		self.msg = Twist()

		self.msg.linear.x = 0.0
		self.msg.linear.y = 0.0
		self.msg.linear.z = 0.0

		self.e0 = np.array([[0.0],[0.0],[0.0]])



		self.q0 = np.array([ [[0.0],[0.0]] , [[0.0],[0.0]], [[0.0],[0.0]] ])
		self.q = np.array([[0.0],[0.0],[0.0]])



		deltaT = 0.1

	    # Matrices para filter
		self.P = np.array([ [[1.0,1.0],[1.0,1.0]] , [[1.0,1.0],[1.0,1.0]] , [[1.0,1.0],[1.0,1.0]] ] )

		self.F = np.array( [ [[1.0,deltaT],[0.0,1.0]] , [[1.0,deltaT],[0.0,1.0]] , [[1.0,deltaT],[0.0,1.0]] ])
		self.G = np.array([ [[deltaT**2/2],[deltaT]],[[deltaT**2/2],[deltaT]],[[deltaT**2/2],[deltaT]] ])

		self.H= np.array( [ [[1.0,0.0]] , [[1.0,0.0]] , [[1.0,0.0]]] )
		self.I = np.array([ [ [1.0,0.0] , [0.0,1.0]] , [ [1.0,0.0] , [0.0,1.0]] , [ [1.0,0.0] , [0.0,1.0]]])

		self.vel_med = np.array([[0.0],[0.0],[0.0]])
		self.pos_med = np.array([[0.0],[0.0],[0.0]])
		self.ace_med = np.array([[0.0],[0.0],[0.0]])

				# self.P = np.array([[[1]],[[1]]])

				# self.F = np.array([[[1],[deltaT]],[[0],[1]]])
				# self.G = np.array([[[deltaT**2/2]],[[deltaT]]])

				# self.H= np.array([[[1]],[[0]]])
				# self.I = np.identity(2)



		rospy.Subscriber("/raw_imu", Imu, self.get_position)


	def get_position(self,data):
		a_x = data.linear_acceleration.x
		a_y = data.linear_acceleration.y
		a_z = data.linear_acceleration.z - 9.81
		#print a_x

		q_ddot = np.array([[a_x],[a_y],[a_z]])

		#print q_ddot

		self.vel_med = q_ddot + self.vel_med*0.1
		self.pos_med = self.vel_med + self.pos_med*0.1


		
		Q = 0.02*np.random.randn(3,2,2)
		R = 0.02*np.random.randn(3,1,1)
 		q1 = np.array([[[0.0],[0.0]],[[0.0],[0.0]],[[0.0],[0.0]]])

		q0 =self.q0
			#print Q

		for i in range(3):
			
			# Prediction

			X_k = np.dot(self.F[i], q0[i]) + self.G[i] * q_ddot[i]
			

					#print('F= ', self.F[i].shape , ' G =  ', self.G[i].shape, '   q dd = ',q_ddot[i].shape, '   X_k  ', X_k.shape)
			P_k = np.dot(np.dot(self.F[i] , self.P[i]), self.F[i].T )+ Q[i]
					#print('q0[i]=  ', q0[i].shape )
					#print('P_k = ', P_k.shape, ' H_i = ', self.H[i].T.shape , '  R[i] ' , R[i].shape)
			# Observation
			y_k = self.pos_med[i] - np.dot(self.H[i],X_k)
			K_k  = np.dot(np.dot(P_k,self.H[i].T),np.linalg.inv(np.dot(self.H[i],np.dot(P_k,self.H[i].T)) + R[i]))

			# Update

					#K_k = np.dot(P_k ,np.dot(self.H[i].T ,  np.linalg.inv(R[i])))
					#print(' = ', K_k.shape, ' y_k = ', y_k.shape, ' q0 = ', q0[i].shape)
					#print('G', (self.G).shape)

			q1[i] = q0[i] + np.dot(K_k,y_k)
					#print  q0[i]
					#print(q1)
					#print self.P[i]

			self.P[i] = np.dot((self.I[i] - np.dot(K_k,self.H[i])),P_k)

		
 		self.q0 = q1
 		self.q = np.array([ [q1[0][0][0]],[q1[1][0][0]], [q1[2][0][0]] ])

 		print self.q



	def control2goal(self,Kp,Kd,Ki,goal):
		e = goal - self.q
		e_dot = (self.q - self.e0)/0.1
		e_int = e + self.e0 * 0.1


		U = np.dot(Kp,e) + np.dot(Kd,e_dot) + np.dot(Ki , e_int)
		self.msg.linear.x = U[0].astype(float)
		self.msg.linear.y = U[1].astype(float)
		self.msg.linear.z = U[2].astype(float)


		
		self.cmd.publish(self.msg)
		
		#print U
		self.e0 = e
				#print U

if __name__ == '__main__':
	goal = np.array([[0.0],[0.0],[12.0]])
	Kp = np.diag(np.array([0.3,0.3,0.3]))
	Kd = np.diag(np.array([0.2,0.2,0.4]))
	Ki = np.diag(np.array([0.2,0.2,0.2]))
	drone = Drone()

	while not rospy.is_shutdown():
	    rospy.init_node('position', anonymous=True)

	    rate = rospy.Rate(10) # 10 hz

	    drone.control2goal(Kp,Kd,Ki,goal)




	# q_ddot = np.array([a_x,a_y,a_z])
	

	# Q = np.random.randn(2,1).reshape(1,1,2)
	# R = np.random.randn(2,1).reshape(1,1,2)

	# q_ddot =q_ddot.reshape(1,1,3)
	
	# q0 =self.q0.reshape(1,1,3)
	# #q=self.q.reshape(1,1,3)
	# # Prediction

	# q_k = self.F * q0 + self.G * q_ddot
	# P_k = self.F * self.P* self.F.T + Q
	# print('P_k = ', P_k.shape, ' q_k = ', q_k.shape )
	# # Observation
	# y_k =q_ddot - self.H*q_k

	# # Update

	# K_k = P_k *self.H.T * R
	# print('G', (self.G).shape)
	# print('K_k = ', K_k.shape, ' y_k = ', y_k.shape, ' q0 = ', q0.shape)

	# q = q0 + K_k*y_k

	# self.P = (self.I - K_k*self.H)*P_k

		# 	self.q0 = q.reshape(3,1)
	
		# 	self.q = q.reshape(3,1)

	#self.q = kalman_filter(q_ddot, self.q0, self.q)

####### TRYING ############

