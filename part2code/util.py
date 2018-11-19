import numpy
import math
import random
import sys

FLOAT_EPS = numpy.finfo(numpy.float).eps

BOUNDARY_MIN_X = -10
BOUNDARY_MIN_Y = -10

BOUNDARY_MAX_X = 10
BOUNDARY_MAX_Y = 10

BOUNDARY_MIN_Z = 0
BOUNDARY_MAX_Z = 3

NUM_SAMPLES = 300
FIXED_K = 10


def print_board(board):
    board.reverse()
    for row in board:
        print " ".join(row)
    print "\n"


def pathFromGoal(vertex, start):
    path = []
    while True:
        path.append(vertex)
        if vertex == start:
            break
        vertex = vertex.parent
    return path[::-1]


def asymptotic_k(num_nodes):
    return int(math.e * 1 + (1.0 / 4.0) * math.log(num_nodes))


def Bresenham3D(node1, node2):
    x1, y1, z1 = node1.getX() - BOUNDARY_MIN_X, node1.getY() - BOUNDARY_MIN_Y, node1.getZ()
    x2, y2, z2 = node2.getX() - BOUNDARY_MIN_X, node2.getY() - BOUNDARY_MIN_Y, node2.getZ()

    points = []

    points.append((x1 + BOUNDARY_MIN_X, y1 + BOUNDARY_MIN_Y, z1))
    points.append((x2 + BOUNDARY_MIN_X, y2 + BOUNDARY_MIN_Y, z2))

    diffX = x2 - x1
    diffY = y2 - y1
    diffZ = z2 - z1

    biggest = max(abs(diffX), abs(diffY), abs(diffZ))

    dirX = 0.0
    dirY = 0.0
    dirZ = 0.0

    if biggest != 0:
        dirX = diffX / biggest
        dirY = diffY / biggest
        dirZ = diffZ / biggest

    currentX = x1
    currentY = x2
    currentZ = z1

    while True:
        currentX += dirX
        currentY += dirY
        currentZ += dirZ
        points.append([currentX + BOUNDARY_MIN_X, currentY + BOUNDARY_MIN_Y, currentZ])

        if currentX > BOUNDARY_MAX_X or currentY > BOUNDARY_MAX_Y or currentZ > BOUNDARY_MAX_Z:
            break
        elif currentX < BOUNDARY_MAX_X or currentY < BOUNDARY_MIN_Y or currentZ < BOUNDARY_MIN_Z:
            break

        if not (abs(currentX - x2) > 1 or abs(currentY - y2) > 1 or abs(currentZ - z2) > 1):
            break

    return points

def bresenham(x0, y0, x1, y1):
    """Yield integer coordinates on the line from (x0, y0) to (x1, y1).
    Input coordinates should be integers.
    The result will contain both the start and the end point.
    """
    dx = x1 - x0
    dy = y1 - y0

    xsign = 1 if dx > 0 else -1
    ysign = 1 if dy > 0 else -1

    dx = abs(dx)
    dy = abs(dy)

    if dx > dy:
        xx, xy, yx, yy = xsign, 0, 0, ysign
    else:
        dx, dy = dy, dx
        xx, xy, yx, yy = 0, ysign, xsign, 0

    D = 2*dy - dx
    y = 0

    for x in range(dx + 1):
        yield x0 + x*xx + y*yx, y0 + x*xy + y*yy
        if D >= 0:
            y += 1
            D -= 2*dx
        D += 2*dy

def distance(in_s, in_t):  # calculates the offsets of each dimension
    sum_d = math.pow(in_s.getX() - in_t.getX(), 2.0)
    sum_d += math.pow(in_s.getY() - in_t.getY(), 2.0)
    sum_d += math.pow(in_s.getZ() - in_t.getZ(), 2.0)

    #  sum_d += math.fabs(in_s.rotation.distance_quaternion(in_t.rotation))

    if sum_d == 0:
        return sys.maxsize

    return math.sqrt(sum_d)

def distance_controls(in_s, in_t): # calculates the offsets of each dimension
    sum_d = math.pow(in_s.getX() - in_t.getX(), 2.0)
    sum_d += math.pow(in_s.getY() - in_t.getY(), 2.0)

    if sum_d == 0:
        return sys.maxsize

    return math.sqrt(sum_d)

"""
def rand_quaternion():
    # numpy.random() returns random floats in the half-open interval [0.0, 1.0)
    s = random.uniform(0, 1)
    theta_1 = math.sqrt(1 - s)
    theta_2 = math.sqrt(s)
    phi_1 = 2 * math.pi * random.uniform(0, 1)
    phi_2 = 2 * math.pi * random.uniform(0, 1)
    w = numpy.cos(phi_2) * theta_2
    x = numpy.sin(phi_1) * theta_1
    y = numpy.cos(phi_1) * theta_1
    z = numpy.sin(phi_2) * theta_2

    return Quaternion(x, y, z, w)
"""

def translation_random():
    return translation_matrix_delta(random.uniform(BOUNDARY_MIN_X, BOUNDARY_MAX_X),
                                    random.uniform(BOUNDARY_MIN_Y, BOUNDARY_MAX_Y),
                                    random.uniform(BOUNDARY_MIN_Z, BOUNDARY_MAX_Z))


def translation_ground_zero_random():
    return translation_matrix_delta(random.uniform(BOUNDARY_MIN_X, BOUNDARY_MAX_X),
                                    random.uniform(BOUNDARY_MIN_Y, BOUNDARY_MAX_Y),
                                    0)


def translation_matrix_delta(dx, dy, dz):
    translation = numpy.eye(4)

    translation[0][3] = dx
    translation[1][3] = dy
    translation[2][3] = dz
    return translation


def translation_vertex_delta(dx, dy, dz):
    return [dx, dy, dz]


def translation_delta_theta(theta, dx, dy, dz):
    rotational = numpy.eye(4)

    # theta is in degrees do not forget!
    cos_a, cos_b, cos_g = numpy.cos(theta)
    sin_a, sin_b, sin_g = numpy.sin(theta)

    rotational.flat = (
        cos_a * cos_b, cos_a * sin_a * sin_g - sin_a * cos_g, cos_a * sin_b * cos_g + sin_a * sin_g, dx,
        sin_a * sin_b, sin_a * sin_b * sin_g + cos_a * cos_g, sin_a * sin_b * cos_g - cos_a * sin_g, dy,
        -sin_b, cos_b * sin_g, cos_b * cos_g, dz,
        0, 0, 0, 1
    )
    return rotational

def random_theta():
    theta = random.uniform(0, math.pi)
    return theta