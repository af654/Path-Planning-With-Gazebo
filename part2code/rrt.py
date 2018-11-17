"""
Path Planning with Kinodynamic Randomized Rapidly-Exploring Random Trees (RRT) for Ackermann vehicle movement
Goal: Get a big enough RRT tree of nodes for the Ackermann robot to find a path from the bottom left to top right of the maze
Nodes represent controls of the Ackermann vehicle
"""

import numpy as np
import matplotlib.pyplot as plt
import math
import random

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
	self.linear = 0
	self.angular = 0

    def getX(self):
        return self.translation[0][3]

    def getY(self):
        return self.translation[1][3]

    def getZ(self):
        return self.translation[2][3]

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

class Quaternion:

    def __init__(self, x, y, z, w):
        self.x = x
        self.y = y
        self.z = z
        self.w = w

    # distance function that helps determine distance from sample point a to b in a neighborhood of collision free
    # points
    def distance_quaternion(self, q2):
        return 1 - math.pow(2, self.dot(q2))

    def dot(self, other):
        return self.x * other.x + self.y * other.y + self.z * other.z + self.w * other.w

    def mult(self, other):
        self.x = (other.w * self.x + other.x * self.w + other.y * self.z - other.z * self.y)
        self.y = (other.w * self.y - other.x * self.z + other.y * self.w + other.z * self.x)
        self.z = (other.w * self.z + other.x * self.y - other.y * self.x + other.z * self.w)
        self.w = (other.w * self.w - other.x * self.x - other.y * self.y - other.z * self.z)
        pass

    def as_rotation_matrix(self):
        w, x, y, z = self.w, self.x, self.y, self.z
        Nq = w * w + x * x + y * y + z * z
        if Nq < util.FLOAT_EPS:
            return numpy.eye(3)
        s = 2.0 / Nq
        X = x * s
        Y = y * s
        Z = z * s
        wX = w * X
        wY = w * Y
        wZ = w * Z
        xX = x * X
        xY = x * Y
        xZ = x * Z
        yY = y * Y
        yZ = y * Z
        zZ = z * Z
        return numpy.array(
            [[1.0 - (yY + zZ), xY - wZ, xZ + wY],
             [xY + wZ, 1.0 - (xX + zZ), yZ - wX],
             [xZ - wY, yZ + wX, 1.0 - (xX + yY)]])
      
# class that defines a node in se(3)
class RelativePosition:

    def __init__(self, translation, rotation):
        # need 7 dimensional node because of quaternion definition

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
            points = numpy.dot(points, trans)
        return points

class RoadMap:
    def __init__(self, tree):
        self.graph = nn.NearestNeighbors(util.distance)

        tree.tree = self
        tree.populate()
        tree.connect(self.graph.nr_nodes, self.graph.nodes)

        self.tree = tree
	
class RRTtree(start, goal):
  def __init__(self, start, goal):
        # need 7 dimensional node because of quaternion definition
        self.start = start
	add_sample_point(start)
        self.goal = goal
	
  def populate():
	previous_node = self.start
	for i in range(0,nmax):
		#go through controls
		save_model_state(previous_node)
		
		translation, rotation = self.get_sample_point(previous_node)
		new_node = Node(translation, rotation)
		self.add_sample_point(new_node)
		previous_node = new_node
		
		save_model_state(new_node)
		
	add_sample_point(self.goal)
	
  #get the nearest sample point to the previous point (xnew based on xnear)
  def get_sample_point(previous_node):
	#need a new translation method that translates according to the angle and time
	translation = util.get_translation()
	rotation = util.rand_quaternion()
	return translation, rotation

  def add_sample_point(self, node):
	if node in self.tree.graph.nodes:
            return
        self.tree.graph.add_node(node)
        
  #expand a random point
  #calls subroutines to find nearest node and connect it
  def expand (self):
	#add random node
	x = random.uniform (E.xmin, E.xmax)
	y = random.uniform (E.ymin, E.ymax)
	theta = random.uniform (0, math.pi)
	n= self.number_of_nodes() #new node number
	self.add_node(n,x,y,theta)
 
        if E.isfree()!=0:
            #find nearest node
	    nnear = self.near(n)
	    #find new node based on the transition equation
	    self.step(nnear,n)

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
  start = Node(util.translation_matrix_delta(0, 3, 0), util.rand_quaternion())
  goal = Node(util.translation_matrix_delta(5, 5, 0), Quaternion(0, 0, 0, 0))
  
  #create an RRT tree with a start node
  rrt_tree=Road_Map(RRTtree(start, goal))
  
  controls_of_ackermann = rrt_tree
  #run a star on the tree to get solution path
  controls_in_path = rrt_tree.findPath(start, goal)
  
  send_to_gazebo(controls_of_ackermann, controls_in_path)
  #each rrt node in the tree has a translation and rotation
  #this translates to a pose and a twist for the ackermann vehicle model
  
  
