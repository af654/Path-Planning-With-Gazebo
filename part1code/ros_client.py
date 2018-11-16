import rospy
from pqp_server.srv import *

def send_path(start, goal):
    rospy.wait_for_service('pqp_server')
    try:
        pqp_server = rospy.ServiceProxy('pqp_server', pqpRequest)
        result = pqp_server(T, R)
        return result
    except rospy.ServiceException, e:
        print "Service Call Failed: %s" % e
