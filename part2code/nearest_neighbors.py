import random
import math
import sys

MAX_KK = 2000
INIT_NODE_SIZE = 1000
INIT_CAP_NEIGHBORS = 200


def resort_proximity_nodes(close_nodes, distances, index):
    while index > 0 and distances[index] < distances[index - 1]:
        temp = distances[index]
        distances[index] = distances[index - 1]
        distances[index - 1] = temp

        temp_node = close_nodes[index]
        close_nodes[index] = close_nodes[index - 1]
        close_nodes[index - 1] = temp_node

        index -= 1

    return index


def pad_or_truncate(node_list, length, default):
    return node_list[:length] + [default] * (length - len(node_list))


class NearestNeighbors:

    def __init__(self, distance_function):
        self.nodes = pad_or_truncate([], INIT_NODE_SIZE, None)
        self.second_nodes = pad_or_truncate([], MAX_KK, None)
        self.second_distances = pad_or_truncate([], MAX_KK, sys.maxint)
        self.nr_nodes = 0
        self.cap_nodes = INIT_NODE_SIZE
        self.added_node_id = 0
        self.distance_function = distance_function

    def add_node(self, node):
        k = self.percolation_threshold()
        new_k = self.find_k_close(node, self.second_nodes, self.second_distances, k)

        if self.nr_nodes >= self.cap_nodes - 1:
            self.cap_nodes = 2 * self.cap_nodes
            self.nodes = pad_or_truncate(self.nodes, self.cap_nodes, None)

        self.nodes[self.nr_nodes] = node

        node.set_index(self.nr_nodes)
        self.nr_nodes += 1

        for i in range(0, new_k):
            node.add_neighbor(self.second_nodes[i].get_index())
            self.second_nodes[i].add_neighbor(node.get_index())

    def add_nodes(self, graph_nodes, nr_new_nodes):
        if self.nr_nodes + nr_new_nodes >= self.cap_nodes - 1:
            self.cap_nodes = self.nr_nodes + nr_new_nodes + 10
            self.nodes = pad_or_truncate(self.nodes, self.cap_nodes, None)
        for node_head_index in range(0, nr_new_nodes):
            k = self.percolation_threshold()
            new_k = self.find_k_close(graph_nodes[node_head_index], self.second_nodes, self.second_distances, k)

            self.nodes[self.nr_nodes] = graph_nodes[node_head_index]
            graph_nodes[node_head_index].set_index(self.nr_nodes)
            self.nr_nodes += 1

            for j in range(0, new_k):
                graph_nodes[node_head_index].add_neighbor(self.second_nodes[j].get_index())
                self.second_nodes[j].add_neighbor(graph_nodes[node_head_index].get_index())

    def remove_node(self, graph_node):
        nr_neighbors, neighbors = graph_node.get_neighbors()
        for i in range(0, nr_neighbors):
            self.nodes[neighbors[i]].delete_neighbor(graph_node.get_index())

        index = graph_node.get_index()
        if index < self.nr_nodes - 1:
            self.nodes[index] = self.nodes[self.nr_nodes - 1]
            self.nodes[index].set_index(index)

            nr_neighbors, neighbors = self.nodes[index].get_neighbors()
            for i in range(0, nr_neighbors):
                self.nodes[neighbors[i]].replace_neighbor(self.nr_nodes - 1, index)
        self.nr_nodes -= 1
        if self.nr_nodes < (self.cap_nodes - 1) / 2:
            self.cap_nodes /= 2
            self.nodes = pad_or_truncate(self.nodes, self.cap_nodes, None)

    def average_valence(self):
        all_neighs = 0.0
        for i in range(0, self.nr_nodes):
            all_neighs += self.nodes[i].nr_neighbors
        all_neighs /= self.nr_nodes

    def find_closest(self, state):
        return self.basic_closest_search(state)

    def find_k_close(self, state, close_nodes, distances, k):
        if self.nr_nodes == 0:
            return 0

        if k > MAX_KK:
            k = MAX_KK
        elif k >= self.nr_nodes:
            for i in range(0, self.nr_nodes):
                close_nodes[i] = self.nodes[i]
                distances[i] = self.distance_function(self.nodes[i], state)
            self.sort_proximity_nodes(close_nodes, distances, 0, self.nr_nodes - 1)
            return self.nr_nodes

        self.clear_added()

        closest_search_set = self.basic_closest_search(state)

        distances[0] = closest_search_set[0]
        min_index = closest_search_set[1]
        close_nodes[0] = closest_search_set[2]

        self.nodes[min_index].added_index = self.added_node_id

        min_index = 0
        nr_elements = 1

        while True:
            nr_neighbors, neighbors = self.nodes[close_nodes[min_index].get_index()].get_neighbors()
            lowest_replacement = nr_elements

            for j in range(0, nr_neighbors):
                the_neighbor = self.nodes[neighbors[j]]
                if not self.does_node_exist(the_neighbor):
                    the_neighbor.added_index = self.added_node_id

                    distance = self.distance_function(the_neighbor, state)
                    to_resort = False

                    if nr_elements < k:
                        close_nodes[nr_elements] = the_neighbor
                        distances[nr_elements] = distance
                        nr_elements += 1
                        to_resort = True
                    elif distance < distances[k - 1]:
                        close_nodes[k - 1] = the_neighbor
                        distances[k - 1] = distance
                        to_resort = True

                    if to_resort:
                        test = resort_proximity_nodes(close_nodes, distances, nr_elements - 1)
                        lowest_replacement = 0
                        if test < lowest_replacement:
                            lowest_replacement = test
                        else:
                            lowest_replacement = lowest_replacement

            if min_index < lowest_replacement:
                min_index += 1
            else:
                min_index = lowest_replacement

            if min_index >= nr_elements:
                break

        return nr_elements

    def find_delta_close_and_closest(self, state, close_nodes, distances, delta):
        if self.nr_nodes == 0:
            return 0

        self.clear_added()

        closest_search_set = self.basic_closest_search(state)

        distances[0] = closest_search_set[0]
        min_index = closest_search_set[1]
        close_nodes[0] = closest_search_set[2]

        if distances[0] > delta:
            return 1

        self.nodes[min_index].added_index = self.added_node_id

        nr_points = 1

        for counter in range(0, nr_points):
            nr_neighbors, neighbors = close_nodes[counter].get_neighbors()

            for j in range(0, nr_neighbors):
                the_neighbor = self.nodes[neighbors[j]]
                if not self.does_node_exist(the_neighbor):
                    the_neighbor.added_index = self.added_node_id
                    distance = self.distance_function(the_neighbor, state)
                    if distance < delta and nr_points < MAX_KK:
                        close_nodes[nr_points] = the_neighbor
                        distances[nr_points] = distance
                        nr_points += 1

        if nr_points > 0:
            self.sort_proximity_nodes(close_nodes, distances, 0, nr_points - 1)
        return nr_points

    def find_delta_close(self, state, close_nodes, distances, delta):
        if self.nr_nodes == 0:
            return 0

        self.clear_added()

        closest_search_set = self.basic_closest_search(state)

        distances[0] = closest_search_set[0]
        min_index = closest_search_set[1]
        close_nodes[0] = closest_search_set[2]

        if distances[0] < delta:
            return 0

        self.nodes[min_index].added_index = self.added_node_id

        nr_points = 1
        nr_neighbors = 0
        neighbors = None
        for counter in range(0, nr_points):
            nr_neighbors, neighbors = close_nodes[counter].get_neighbors()

        for j in range(0, nr_neighbors):
            the_neighbor = self.nodes[neighbors[j]]
            if not self.does_node_exist(the_neighbor):
                the_neighbor.added_index = self.added_node_id
                distance = self.distance_function(the_neighbor, state)
                if distance < delta and nr_points < MAX_KK:
                    close_nodes[nr_points] = the_neighbor
                    distances[nr_points] = distances
                    nr_points += 1

        if nr_points > 0:
            self.sort_proximity_nodes(close_nodes, distances, 0, nr_points - 1)

        return nr_points

    def sort_proximity_nodes(self, close_nodes, distances, low, high):
        if low < high:
            pivot_distance = distances[low]
            pivot_node = close_nodes[low]

            left = low
            right = high

            while left < right:
                while low <= high and distances[left] <= pivot_distance:
                    left += 1
                while distances[right] > pivot_distance:
                    right -= 1

                if left < right:
                    temp = distances[left]
                    distances[left] = distances[right]
                    distances[right] = temp

                    temp_node = close_nodes[left]
                    close_nodes[left] = close_nodes[right]
                    close_nodes[right] = temp_node

            distances[low] = distances[right]
            distances[right] = pivot_distance

            close_nodes[low] = close_nodes[right]
            close_nodes[right] = pivot_node

            self.sort_proximity_nodes(close_nodes, distances, low, right - 1)
            self.sort_proximity_nodes(close_nodes, distances, right + 1, high)

    def does_node_exist(self, query_node):
        return query_node.added_index == self.added_node_id

    def basic_closest_search(self, state):
        if self.nr_nodes == 0:
            return None

        nr_samples = self.sampling_function()
        min_distance = 2147483647

        min_index = -1
        index = 0
        for i in range(0, nr_samples):
            index = random.randint(0, self.nr_nodes - 1)
            distance = self.distance_function(self.nodes[index], state)
            if distance < min_distance:
                min_distance = distance
                min_index = index

        while True:
            old_min_index = min_index
            nr_neighbors, neighbors = self.nodes[min_index].get_neighbors()
            for j in range(0, nr_neighbors):
                distance = self.distance_function(self.nodes[index], state)
                if distance < min_distance:
                    min_distance = distance
                    min_index = neighbors[j]
            if old_min_index == min_index:
                break

        the_distance = min_distance
        the_index = min_index

        return the_distance, the_index, self.nodes[min_index]

    def clear_added(self):
        self.added_node_id += 1

    def add_node_deserialize(self, graph_node):
        if self.nr_nodes >= self.cap_nodes + 1:
            self.cap_nodes *= 2
            self.nodes = pad_or_truncate(self.nodes, self.cap_nodes, None)
        self.nodes[self.nr_nodes] = graph_node
        self.nr_nodes += 1

    def get_node_by_index(self, index):
        for i in range(0, self.nr_nodes):
            if self.nodes[i][0].get_index() == index:
                return self.nodes[i][0]
        return None

    def sampling_function(self):
        if self.nr_nodes < 500:
            return self.nr_nodes / 5 + 1
        else:
            return 100 + self.nr_nodes / 500

    def percolation_threshold(self):
        if self.nr_nodes > 12:
            return int((3.5 * math.log(self.nr_nodes)))
        else:
            return self.nr_nodes
