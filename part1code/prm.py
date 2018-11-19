from abc import abstractmethod

import numpy
import math
import sys
import heapq
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D

import util
import nearest_neighbors as nn
import pqp_ros_client_ours as pqp
from gazebo_msgs.msg import ModelState, ModelStates
from geometry_msgs.msg import Point, Pose, Twist, Quaternion
import rospy
import roslib
from std_msgs.msg import String
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState
from geometry_msgs.msg import PoseWithCovarianceStamped
from std_msgs.msg import Empty as EmptyMsg
from std_srvs.srv import Empty as EmptySrv
from datetime import datetime

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


class RoadMap:
    def __init__(self, sampler):
        self.graph = nn.NearestNeighbors(util.distance)

        sampler.road_map = self
        sampler.sample()
        sampler.connect(self.graph.nr_nodes, self.graph.nodes)

        self.sampler = sampler

        # TODO build k neighbors here and populate nodes.


@abstractmethod
class PRMSample:

    def __init__(self):
        self.road_map = None
        pass

    def sample(self):
        for n in range(0, util.NUM_SAMPLES):
            translation, rotation = self.get_sample_point()  # TODO we need to check if the list contains a neighbor in the same position or else we got problems
            node = Node(translation, rotation)
            if pqp.pqp_client(node.as_translation_vector(), node.rotation.as_rotation_matrix().flatten()):
                self.add_node(node)
        pass

    @abstractmethod
    def connect(self, nr_nodes, nodes):
        pass

    @staticmethod
    def get_sample_point():
        translation = util.translation_random()
        rotation = util.rand_quaternion()
        return translation, rotation

    def add_node(self, node):
        if node in self.road_map.graph.nodes:
            return
        self.road_map.graph.add_node(node)


class FixedKPRM(PRMSample):

    def __init__(self):
        PRMSample.__init__(self)

    def connect(self, nr_nodes, nodes):
        graph = self.road_map.graph  # type: nn.NearestNeighbors

        close_neighbors = nn.pad_or_truncate([], util.FIXED_K, -1)
        neighbor_distance = nn.pad_or_truncate([], util.FIXED_K, sys.maxint)

        for parent_index in range(0, nr_nodes):
            node = nodes[nr_nodes - parent_index - 1]

            num_neighbors = graph.find_k_close(node, close_neighbors, neighbor_distance, util.FIXED_K)

            if node in close_neighbors:
                num_neighbors -= 1
                for index in range(0, num_neighbors):
                    close_neighbors[index] = close_neighbors[index + 1]

            for index in range(0, num_neighbors):
                neighbor = close_neighbors[index]
                points_list = util.Bresenham3D(node, neighbor)

                collision = False

                for points in points_list:
                    if not pqp.pqp_client(points, neighbor.rotation.as_rotation_matrix().flatten()):
                        collision = True
                        break
                if collision:
                    close_neighbors[index] = None  # We need to build this list later for none elements

            nr_neighbors = 0
            for i in range(0, num_neighbors):
                neighbor = close_neighbors[i]
                if neighbor is None:
                    continue
                node.neighbors[nr_neighbors] = neighbor.get_index()
                nr_neighbors += 1

            node.nr_neighbors = nr_neighbors

    pass


class ConnectedComponentPRM(PRMSample):

    def __init__(self):
        PRMSample.__init__(self)
        self.open_set = dict()

    def connect(self, nr_nodes, nodes):
        self.open_set = dict()
        graph = self.road_map.graph  # type: nn.NearestNeighbors

        close_neighbors = nn.pad_or_truncate([], util.FIXED_K, None)
        neighbor_distance = nn.pad_or_truncate([], util.FIXED_K, sys.maxint)

        for parent_index in range(0, nr_nodes):
            node = nodes[nr_nodes - parent_index - 1]

            num_neighbors = graph.find_k_close(node, close_neighbors, neighbor_distance, util.FIXED_K)
            if node in close_neighbors:
                num_neighbors -= 1
                for index in range(0, num_neighbors):
                    close_neighbors[index] = close_neighbors[index + 1]

            for index in range(0, num_neighbors):
                neighbor = close_neighbors[index]

                count = self.open_set.get(neighbor, 0)
                if count > 3:
                    close_neighbors[index] = None
                    continue

                self.open_set.update({neighbor: count + 1})

                points_list = util.Bresenham3D(node, neighbor)

                collision = False

                for points in points_list:
                    if not pqp.pqp_client(points, neighbor.rotation.as_rotation_matrix().flatten()):
                        collision = True
                        break
                if collision:
                    close_neighbors[index] = None  # We need to build this list later for none elements

            nr_neighbors = 0
            for i in range(0, num_neighbors):
                neighbor = close_neighbors[i]
                if neighbor is None:
                    continue
                node.neighbors[nr_neighbors] = neighbor.get_index()
                nr_neighbors += 1

            node.nr_neighbors = nr_neighbors

        pass


class AsymptoticPRM(FixedKPRM):

    def __init__(self):
        FixedKPRM.__init__(self)

    def sample(self):
        FixedKPRM.sample(self)

    def connect(self, nr_nodes, nodes):
        graph = self.road_map.graph  # type: nn.NearestNeighbors

        max_k = util.asymptotic_k(graph.nr_nodes)
        close_neighbors = nn.pad_or_truncate([], max_k, None)
        neighbor_distance = nn.pad_or_truncate([], max_k, sys.maxint)

        for parent_index in range(0, nr_nodes):
            node = nodes[nr_nodes - parent_index - 1]

            num_neighbors = graph.find_k_close(node, close_neighbors, neighbor_distance, util.asymptotic_k(parent_index
                                                                                                           + 1))

            if node in close_neighbors:
                num_neighbors -= 1
                for index in range(0, num_neighbors):
                    close_neighbors[index] = close_neighbors[index + 1]

            for index in range(0, num_neighbors):
                neighbor = close_neighbors[index]
                points_list = util.Bresenham3D(node, neighbor)

                collision = False

                for points in points_list:
                    if not pqp.pqp_client(points, neighbor.rotation.as_rotation_matrix().flatten()):
                        collision = True
                        break
                if collision:
                    close_neighbors[index] = None  # We need to build this list later for none elements

            nr_neighbors = 0
            for i in range(0, num_neighbors):
                neighbor = close_neighbors[i]
                if neighbor is None:
                    continue
                node.neighbors[nr_neighbors] = neighbor.get_index()
                nr_neighbors += 1

            node.nr_neighbors = nr_neighbors


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

        return util.distance(vertex, goal) * 1.222222

    @staticmethod
    def c(from_vertex, to_vertex):  # the straight line distance between the s node and e node.
        return util.distance(from_vertex, to_vertex)

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
        self.road_map.sampler.connect(2, end_points)

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


class FDAPath(APath):

    def updateVertex(self, vertex, neighbor, goal):
        if vertex.parent.edgeCost + Path.c(vertex.parent, neighbor) < neighbor.edgeCost:
            neighbor.edgeCost = vertex.parent.edgeCost + Path.c(vertex.parent, neighbor)
            neighbor.parent = vertex.parent
            if neighbor in self.openSet:
                self.remove(neighbor)
            self.f(neighbor, goal)
            self.add(neighbor)


# TODO: get sample from the roadmap and check if collision free
def send_to_gazebo(verticies_in_path, path):
    print "hello world\n"
    rospy.init_node('move_robot_to_given_place')
    print "hello world 2\n"

    counter = 0
    while not rospy.is_shutdown():
        if counter >= len(verticies_in_path):
            break

        rospy.sleep(1)
        print "reposition robot now"

        reposition_robot(verticies_in_path[counter])
        counter += 1


def reposition_robot(vertex):
    state_pub = rospy.Publisher("/gazebo/set_model_state", ModelState, queue_size=10)

    pose = Pose()

    pose.position.x = vertex.getX()
    pose.position.y = vertex.getY()
    pose.position.z = vertex.getZ()

    pose.orientation.x = vertex.rotation.x
    pose.orientation.y = vertex.rotation.y
    pose.orientation.z = vertex.rotation.z
    pose.orientation.w = vertex.rotation.w

    state = ModelState()

    state.model_name = "piano2"
    state.reference_frame = "world"
    state.pose = pose

    state_pub.publish(state)


def main():
    startTime = datetime.now()
    road_map = RoadMap(FixedKPRM())
    fda = APath(road_map)
    graph = road_map.graph

    start = Node(util.translation_matrix_delta(1, 6, 0), util.rand_quaternion())
    goal = Node(util.translation_matrix_delta(4, 11, 0), Quaternion(0, 0, 0, 0))

    verticies_in_path = fda.findPath(start, goal)
    path = map(lambda vertex: (vertex.getX(), vertex.getY(), vertex.getZ()), verticies_in_path)

    fig = pyplot.figure()
    ax = fig.add_subplot(111, projection='3d')

    x_paths = list(map(lambda coord: coord[0], path))
    y_paths = list(map(lambda coord: coord[1], path))
    z_paths = list(map(lambda coord: coord[2], path))

    x_samples = []
    y_samples = []
    z_samples = []

    neighbor_set = set()

    traversed_sum = 0
    node_prev = None

    for node in verticies_in_path:
        traversed_sum += node.nr_neighbors

        print node.f
        if node_prev is not None:
            pyplot.plot([node.getX(), node_prev.getX()], [node.getY(), node_prev.getY()],
                        [node.getZ(), node_prev.getZ()], 'ro-', color='yellow')

        node_prev = node

        for index in range(0, node.nr_neighbors):
            neighbor = graph.nodes[node.neighbors[index]]

            pyplot.plot([neighbor.getX(), node.getX()], [neighbor.getY(), node.getY()],
                        [neighbor.getZ(), node.getZ()], 'ro-', color='purple')
            if neighbor in neighbor_set:
                continue
            neighbor_set.add(neighbor)

            x_samples.append(neighbor.getX())
            y_samples.append(neighbor.getY())
            z_samples.append(neighbor.getZ())

    print "Number of neighbor's expected: "

    expected_sum = 0
    for index in range(0, graph.nr_nodes):
        expected_sum += util.asymptotic_k(index + 1)

    print expected_sum
    print "\n"

    print "Number of unique neighbors in path"
    print traversed_sum
    print "\n"

    ax.scatter(x_samples, y_samples, z_samples, color='orange', marker='o', zorder=15)
    ax.scatter(x_paths, y_paths, z_paths, color='green', marker='o', zorder=10)
    ax.scatter([start.getX()], [start.getY()], [start.getZ()], color='red', marker='o', zorder=5)
    ax.scatter([goal.getX()], [goal.getY()], [goal.getZ()], color='blue', marker='o', zorder=0)

    print datetime.now() - startTime 
    pyplot.show()

    send_to_gazebo(verticies_in_path, path)
    pass


if __name__ == "__main__":
    main()
