"""
Path Planning with Kinodynamic Randomized Rapidly-Exploring Random Trees (RRT) for Ackermann vehicle movement
Goal: Get a big enough RRT tree of nodes for the Ackermann robot to find a path from the bottom left to top right of the maze
Nodes represent controls of the Ackermann vehicle
"""

import numpy as np
import matplotlib.pyplot as plt
import math
import random
import util
import nearest_neighbors as nn
#import pqp_ros_client_ours as pqp
from gazebo_msgs.msg import ModelState, ModelStates
from geometry_msgs.msg import Point, Pose, Twist
import rospy
from std_msgs.msg import String
from gazebo_msgs.srv import SetModelState
from std_msgs.msg import Empty as EmptyMsg
from std_srvs.srv import Empty as EmptySrv

#node limit
nmax = 2000
SPEEDS = [-1,0,1]

class Node(RelativePosition):
    def __init__(self, translation, rotation):
        RelativePosition.__init__(self, translation, rotation)
        self.neighbors = nn.pad_or_truncate([], nn.INIT_CAP_NEIGHBORS, -1)
        self.nr_neighbors = 0
        self.added_index = 0
        self.index = 0
        self.cap_neighbors = nn.INIT_CAP_NEIGHBORS
	
        self.parent = self
        self.edgeCost = 0
        self.f = 0
	
	    self.speed = 0
        #linear position
	    self.theta = 0
        #angular position
        self.angular = 0

    def getX(self):
        return self.translation[0][2]

    def getY(self):
        return self.translation[1][2]

    def set_index(self, index):
        self.index = index

    def get_index(self):
        return self.index

    def get_neighbors(self):
        return self.nr_neighbors, self.neighbors

    def add_neighbor(self, nd):
        if self.nr_neighbors >= self.cap_neighbors - 1:
            self.cap_neighbors = 2 * self.cap_neighbors
            self.neighbors = nn.pad_or_truncate(self.neighbors, self.cap_neighbors, -1)
        self.neighbors[self.nr_neighbors] = nd
        self.nr_neighbors += 1

    def delete_neighbor(self, nd):
        index = 0
        for x in range(0, self.nr_neighbors):
            if self.neighbors[index] == nd:
                break
            index += 1
        if index >= self.nr_neighbors:
            return

        for i in range(index, self.nr_neighbors):
            self.neighbors[i] = self.neighbors[i + 1]
        self.nr_neighbors -= 1

    def replace_neighbor(self, prev, new_index):
        index = 0
        for x in range(0, self.nr_neighbors):
            if self.neighbors[index] == prev:
                break
            index += 1
        if index >= self.nr_neighbors:
            return
        self.neighbors[index] = new_index

    def as_translation_vector(self):
        return [self.getX(), self.getY(), self.getZ()]

    def reset(self):
        self.edgeCost = 0
        self.parent = self
        self.f = 0

    def __hash__(self):
        return hash((self.getX(), self.getY(), self.getZ()))

    def __eq__(self, other):
        if other is None:
            return False
        return (self.getX(), self.getY(), self.getZ()) == (other.getX(), other.getY(), other.getZ())

    def __cmp__(self, other):
        if other is None:
            return -1
        return cmp((self.f, self.f - sys.maxunicode * self.edgeCost),
                   (other.f, other.f - sys.maxunicode * other.edgeCost))
      
# class that defines a node in se(2)
class RelativePosition:

    def __init__(self, translation, rotation):
        #only a 4 dimensional control space
        #we are feeding linear and angular velocity as the twist to the robot 
        #and x and y as the pose to the robot

        self.translation = translation
        self.rotation = rotation

    @classmethod
    def from_relative_position(cls, relative_position):
        return RelativePosition(relative_position.translation, relative_position.rotation)

    def transform(self, trans):
        return self.transform_multi([trans])

    def transform_multi(self, translations):
        points = self.translation
        for trans in translations:
            points = np.dot(points, trans)
        return points

class RoadMap:
    def __init__(self, tree):
        self.graph = nn.NearestNeighbors(util.distance_controls)

        tree.tree_of_controls = self
        tree.populate()
       # tree.connect(self.graph.nr_nodes, self.graph.nodes)
	
class RRTtree(start, goal):
    def __init__(self, start, goal):
        self.start = start
        self.goal = goal

        #this is here for testing purposes - need to get rid of and add gazebo integration
        self.uSpeed = [2, 1, 0.5, 0.25 ]
		self.uSteer = [math.pi/-6.0,math.pi/-12.0,math.pi/-18.0, 0.0, math.pi/6.0,math.pi/12.0,math.pi/18.0]
		self.ut = 1
	
    #function that populates the rrt with controls
    def populate():
        previous_node = self.start
        for i in range(0,nmax):
            #populate the sample space with a random control with a random duration
            #add random node
		    translation, rotation = self.get_sample_point()
		    new_node = Node(translation, rotation)
		    self.add_sample_point(new_node)

            #find nearest node to random node previous_node
		    previous_node = new_node
            expand(new_node)

            self.remove_sample_point(new_node)
	
    #get the nearest sample point to the previous point (xnew based on xnear)
    def get_sample_point()):
	    translation = get_translation_controls()
	    rotation = rand_quaternion_controls()
	    return translation, rotation

    def add_sample_point(self, node):
	    if node in self.tree.graph.nodes:
            return
        self.tree.graph.add_node(node)

    def remove_sample_point(self, node):
        nodes = self.tree.graph.nodes
        for i in nodes:
            if(node == nn.nodes[i]):
                #TODO: remove random point from the control space
        
    #calls subroutines to find nearest node and connect it
    def expand (self, new_node):
        graph = self.tree.graph  # type: nn.NearestNeighbors
        close_neighbors = nn.pad_or_truncate([], util.FIXED_K, -1)
        neighbor_distance = nn.pad_or_truncate([], util.FIXED_K, sys.maxint)
        #find nearest node
        num_neighbors = graph.find_k_close(node, close_neighbors, neighbor_distance, util.FIXED_K)
        node_near = self.near(num_neighbors, new_node)
	    #find new node to connect nearest to new
	    self.step(node_near,new_node)

    #returns the index of the nearest node within num_neighbors to new_node
    def near(num_neighbors, new_node):
        d_min = self.distance_controls(0,new_node)
        node_near
            for i in num_neighbors:
                if self.distance_controls(i,new_node) < dmin:
                    dmin=self.distance_controls(i,new_node)
                    node_near = i
        return node_near

    #state transition 
	def step(self,nnear,nrand):
		(xn,yn,thetan) = (nnear.x, nnear.y, nnear.theta)
		(xran,yran,thetaran) = (nrand.x, nrand.y, nrand.theta)
		
		#compute all reachable states
		xr=[]
		yr=[]
		thetar=[]
		usp=[]
		ust=[]
		for i in self.uSpeed:
			for j in self.uSteer:
				usp.append(i)
				ust.append(j)
				(x,y,theta)=self.trajectory(xn,yn,thetan,j,i)
				xr.append(x)
				yr.append(y)
				thetar.append(theta)
				
		#find a nearest reachable from nnear to nrand
		dmin = ((((xran-xr[0][-1])**2)+((yran-yr[0][-1])**2))**(0.5))
		near = 0
		for i in range(1,len(xr)):
			d = ((((xran-xr[i][-1])**2)+((yran-yr[i][-1])**2))**(0.5))
			if d < dmin:
				dmin= d
				near = i
        
        #add the control to the control space for sampling 
        add_sample_point(nrand,xr[near][-1],yr[near][-1],thetar[near][-1])
		
def get_translation_controls():
	#generate a random x and y as controls for the translation part
    translation = numpy.eye(2)
    translation[0][2] = random.uniform(-9,10)
    translation[1][2] = random.uniform(-7.5,6.5)
    return translation
	
def rand_quaternion_controls():
    #generate a random angle for the rotation part
    theta = random.uniform(0, math.pi)

def send_to_gazebo(controls_of_ackermann, controls_in_path):
    rospy.init_node('move_robot_to_given_place')
    print "hello world 2\n"

    counter = 0
    while not rospy.is_shutdown():
        if counter >= len(verticies_in_path):
            break

        rospy.sleep(1)
        print "reposition robot now"

        save_model_state(controls_in_path[counter])
        counter += 1
        
def save_model_state(node):
   # Set Gazebo Model pose and twist
    state_pub = rospy.Publisher("/gazebo/set_model_state", ModelState, queue_size=10)
    pose = Pose()
    twist = Twist()

    pose.position.x = node.getX()
    pose.position.y = node.getY()
    pose.position.z = node.getZ()

    pose.orientation.x = node.rotation.x
    pose.orientation.y = node.rotation.y
    pose.orientation.z = node.rotation.z
    pose.orientation.w = node.rotation.w
   
    twist.linear = node.linear
    twist.angular = node.angular

    state = ModelState()

    state.model_name = "ackermann_vehicle"
    state.reference_frame = "world"
    state.pose = pose
    state.twist = twist

    state_pub.publish(state)

def main():
  #start for the robot is the bottom left of the maze and goal is the top right of the maze
  start = Node(util.translation_matrix_delta(-9, -5, 0), Quaternion(0,0,0,0))
  goal = Node(util.translation_matrix_delta(9, 5, 0), Quaternion(0,0,0,0))
  
  #create an RRT tree with a start node
  rrt_tree=RoadMap(RRTtree(start, goal))
  
  controls_of_ackermann = rrt_tree
  #run a star on the tree to get solution path
  controls_in_path = rrt_tree.findPath(start, goal)
  
  send_to_gazebo(controls_of_ackermann, controls_in_path)
  #each rrt node in the tree has a translation and rotation
  #this translates to a pose and a twist for the ackermann vehicle model
  
  
