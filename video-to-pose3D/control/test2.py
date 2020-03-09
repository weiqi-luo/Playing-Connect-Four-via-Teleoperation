import numpy as np
import numpy.linalg as la
 
# @xl_func("numpy_row v1, numpy_row v2: float")
def py_ang(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'    """
    cosang = np.dot(v1, v2)
    sinang = la.norm(np.cross(v1, v2))
    return np.degrees(np.arctan2(sinang, cosang))

print(py_ang((0,1),(1,1)))