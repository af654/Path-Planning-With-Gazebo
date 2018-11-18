"""
Path Planning with Kinodynamic Randomized Rapidly-Exploring Random Trees (RRT) for Ackermann vehicle movement
Goal: Get a big enough RRT tree of nodes for the Ackermann robot to find a path from the bottom left to top right of the maze
Nodes represent controls of the Ackermann vehicle
"""
import heapq
import math
import random
import sys
from abc import abstractmethod

import numpy as np

import nearest_neighbors as nn
import util

# import pqp_ros_client_ours as pqp
# from gazebo_msgs.msg import ModelState, ModelStates
# from geometry_msgs.msg import Point, Pose, Twist
# import rospy
# from std_msgs.msg import String
# from gazebo_msgs.srv import SetModelState

# node limit
nmax = 100

# class that defines a node in se(2)
class RelativePosition:

    def __init__(self, translation, theta):
        # only a 4 dimensional control space
        # we are feeding linear and angular velocity as the twist to the robot
        # and x and y as the pose to the robot

        self.translation = translation
        self.theta = theta

    @classmethod
    def from_relative_position(cls, relative_position):
        return RelativePosition(relative_position.translation, relative_position.theta)

    def transform(self, trans):
        return self.transform_multi([trans])

    def transform_multi(self, translations):
        points = self.translation
        for trans in translations:
            points = np.dot(points, trans)
        return points


class Node(RelativePosition):
    def __init__(self, translation, theta):
        # rotation is just theta since we are sampling only controls
        RelativePosition.__init__(self, translation, theta)
        self.neighbors = nn.pad_or_truncate([], nn.INIT_CAP_NEIGHBORS, -1)
        self.nr_neighbors = 0
        self.added_index = 0
        self.index = 0
        self.cap_neighbors = nn.INIT_CAP_NEIGHBORS

        self.parent = self
        self.edgeCost = 0
        self.f = 0

        # linear position
        self.theta = 0
        # angular position
        self.angular = 0

    def getX(self):
        return self.translation[0][1]

    def getY(self):
        return self.translation[1][1]

    def set_index(self, index):
        self.index = index

    def get_index(self):
        return self.index
    
    def get_theta(self):
        return self.theta

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
        return [self.getX(), self.getY()]

    def reset(self):
        self.edgeCost = 0
        self.parent = self
        self.f = 0

    def __hash__(self):
        return hash((self.getX(), self.getY()))

    def __eq__(self, other):
        if other is None:
            return False
        return ((self.getX(), self.getY()) == (other.getX(), other.getY()))

    def __cmp__(self, other):
        if other is None:
            return -1
        return cmp((self.f, self.f - sys.maxunicode * self.edgeCost),
                   (other.f, other.f - sys.maxunicode * other.edgeCost))


class RoadMap:
    def __init__(self, tree_1):
        self.graph = nn.NearestNeighbors(util.distance_controls)
        self.tree = tree_1

        tree_1.tree = self
        print "we got here"
        tree_1.populate()
        print "we got here too"
    # tree.connect(self.graph.nr_nodes, self.graph.nodes)

class RRTtree():
    def __init__(self, start, goal):
        self.start = start
        self.goal = goal
        self.tree = None
        # this is here for testing purposes - need to get rid of and add gazebo integration
        self.uSpeed = [-1, 0, 1]
        self.uSteer = [math.pi / -6.0, math.pi / -12.0, math.pi / -18.0, 0.0, math.pi / 6.0, math.pi / 12.0,
                       math.pi / 18.0]
        self.ut = 1

    # function that populates the rrt with controls
    # populate the sample space with a random control with a random duration
    def populate(self):
        previous_node = self.start
        for i in range(0, nmax):
            translation, theta = get_translation_controls(), rand_quaternion_controls()
            new_node = Node(translation, theta)
            self.add_sample_point(new_node)
            previous_node = new_node
            self.expand(new_node)
            self.remove_sample_point(new_node)

    # get the nearest sample point to the previous point (xnew based on xnear)

    def get_sample_point(self):
        translation = get_translation_controls()
        theta = rand_quaternion_controls()
        return translation, theta

    def add_sample_point(self, node):
        if node in self.tree.graph.nodes:
            return
        self.tree.graph.add_node(node)

    def remove_sample_point(self, node):
        self.tree.graph.remove_node(node)

    # calls subroutines to find nearest node and connect it
    # find nearest node to random node previous_node
    def expand(self, new_node):
        graph = self.tree.graph  # type: nn.NearestNeighbors
        close_neighbors = nn.pad_or_truncate([], util.FIXED_K, None)
        neighbor_distance = nn.pad_or_truncate([], util.FIXED_K, sys.maxint)
        # find nearest node to new_node based on its neighbors
        num_neighbors = graph.find_k_close(new_node, close_neighbors, neighbor_distance, util.FIXED_K)
        node_near = self.near(close_neighbors,num_neighbors, new_node)
        self.step(node_near, new_node)

    # returns the index of the nearest node within num_neighbors to new_node
    def near(self, close_neighbors,num_neighbors, new_node):
        #find a near node
        translation = np.eye(2)
        translation[0][1] = 0
        translation[1][1] = 0
        theta = 0
        node_zero = Node(translation, theta)
        d_min = util.distance_controls(node_zero, new_node)

        node_near = node_zero
        for i in range(0,num_neighbors):
            node = close_neighbors[i]
            if util.distance_controls(node, new_node) < d_min:
                d_min = util.distance_controls(node, new_node)
                node_near = node

        return node_near

        # state transition
        # find new node to connect nearest to new

    def step(self, nnear, nrand):
        (xn, yn, thetan) = (nnear.getX(), nnear.getY(), nnear.get_theta())
        (xran, yran, thetaran) = (nrand.getX(), nrand.getY(), nrand.get_theta())

        # compute all reachable states
        xr = []
        yr = []
        thetar = []
        usp = []
        ust = []
        for i in self.uSpeed:
            for j in self.uSteer:
                usp.append(i)
                ust.append(j)
                (x, y, theta) = self.trajectory(xn, yn, thetan, j, i)
                xr.append(x)
                yr.append(y)
                thetar.append(theta)

        # find a nearest reachable from nnear to nrand
        dmin = ((((xran - xr[0][-1]) ** 2) + ((yran - yr[0][-1]) ** 2)) ** (0.5))
        near = 0
        for i in range(1, len(xr)):
            d = ((((xran - xr[i][-1]) ** 2) + ((yran - yr[i][-1]) ** 2)) ** (0.5))
            if d < dmin:
                dmin = d
                near = i
        
        #add the control that gets you from nnear to nnew (the nearest reachable)
        translation = np.eye(2)
        translation[0][1] = xr[near][-1]
        translation[1][1] = yr[near][-1]
        theta = thetar[near][-1]
        new_node = Node(translation, theta)

        print("control added to the path", new_node.translation.flatten, new_node.theta)
        self.add_sample_point(new_node)

        #collision detection -> do later

    # generate trajectory by integrating equations of motion
    def trajectory(self, xi, yi, thetai, ust, usp):
        (x, y, theta) = ([], [], [])
        x.append(xi)
        y.append(yi)
        theta.append(thetai)
        dt = 0.01
        for i in range(1, int(self.ut / dt)):
            theta.append(theta[i - 1] + usp * math.tan(ust) / 1.9 * dt)
            x.append(x[i - 1] + usp * math.cos(theta[i - 1]) * dt)
            y.append(y[i - 1] + usp * math.sin(theta[i - 1]) * dt)
        return (x, y, theta)


def get_translation_controls():
    # generate a random x and y as controls for the translation part
    translation = np.eye(2)
    translation[0][1] = random.uniform(-9, 10)
    translation[1][1] = random.uniform(-7.5, 6.5)
    return translation


def rand_quaternion_controls():
    # generate a random angle for the rotation part
    theta = random.uniform(0, math.pi)
    return theta


class Path:

    def __init__(self, road_map):
        self.road_map = road_map
        pass

    @abstractmethod
    def findPath(self, start, goal): raise NotImplementedError

    @abstractmethod
    def updateVertex(self, vertex, neighbor, goal): raise NotImplementedError

    def h(self, vertex, goal):  # the estimated path cost from the node we're at to the goal node

        # return               max(abs(vertex.x - goal.x), abs(vertex.y - goal.y))

        return util.distance_controls(vertex, goal) * 1.222222

    @staticmethod
    def c(from_vertex, to_vertex):  # the straight line distance between the s node and e node.
        return util.distance_controls(from_vertex, to_vertex)

    def f(self, vertex, goal):
        vertex.f = vertex.edgeCost + self.h(vertex, goal)


class APath(Path):

    def __init__(self, graph):
        Path.__init__(self, graph)
        self.heap = []
        self.openSet = set()

    def findPath(self, start, goal):
        graph = self.road_map.graph

        end_points = [start, goal]
        graph.add_nodes(end_points, 2)

        closed = set()

        start.reset()
        goal.reset()

        self.openSet = set()
        self.heap = []
        heapq.heapify(self.heap)

        heapq.heappush(self.heap, start)
        self.openSet.add(start)
        start.f = start.edgeCost + self.h(start, goal)

        while len(self.heap) > 0:
            vertex = heapq.heappop(self.heap)

            if vertex == goal:
                graph.remove_node(start)
                graph.remove_node(goal)
                return util.pathFromGoal(vertex, start)

            closed.add(vertex)

            for index in range(0, vertex.nr_neighbors):
                neighbor = graph.nodes[vertex.neighbors[index]]
                if neighbor not in closed:
                    if neighbor not in self.openSet:
                        neighbor.edgeCost = sys.maxint
                        neighbor.parent = None
                    self.updateVertex(vertex, neighbor, goal)

        graph.remove_node(start)
        graph.remove_node(goal)
        return []

    def updateVertex(self, vertex, neighbor, goal):
        if vertex.edgeCost + Path.c(vertex, neighbor) < neighbor.edgeCost:
            neighbor.edgeCost = vertex.edgeCost + Path.c(vertex, neighbor)
            neighbor.parent = vertex
            if neighbor in self.openSet:
                self.remove(neighbor)
            self.f(neighbor, goal)
            self.add(neighbor)

    def remove(self, vector):
        self.openSet.remove(vector)
        try:
            self.heap.remove(vector)
            heapq.heapify(self.heap)
        except ValueError:
            pass

    def add(self, vector):
        heapq.heappush(self.heap, vector)
        self.openSet.add(vector)
        pass


"""
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
   
    twist.linear = node.theta
    #twist.angular = node.angular

    state = ModelState()

    state.model_name = "ackermann_vehicle"
    state.reference_frame = "world"
    state.pose = pose
    state.twist = twist

    state_pub.publish(state)
"""

def main():
    # start for the robot is the bottom left of the maze and goal is the top right of the maze
    start = Node(util.translation_matrix_delta(-9, -5, 0), util.random_theta())
    goal = Node(util.translation_matrix_delta(9, 5, 0), util.random_theta())

    # create an RRT tree with a start node
    rrt_tree = RoadMap(RRTtree(start, goal))
    a_star = APath(rrt_tree)
    graph = rrt_tree.tree

    controls_of_ackermann = rrt_tree
    # run a star on the tree to get solution path
    controls_in_path = a_star.findPath(start, goal)
    path = map(lambda vertex: (vertex.getX(), vertex.getY()),controls_in_path)
    print path

    # send_to_gazebo(controls_of_ackermann, controls_in_path)

    # each rrt node in the tree has a translation and rotation
    # this translates to a pose and a twist for the ackermann vehicle model

if __name__ == "__main__":
    main()