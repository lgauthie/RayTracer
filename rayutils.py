import numpy as np
#import math as m

def compute_ray_dir(x, y, world):
    # Maybe use something like lookAt here
    inv_width  = (1/float(world.width))
    inv_height = (1/float(world.height))
    a = (2 * ((x + 0.5) * inv_width) - 1) * world.angle * world.aspectratio
    b = (1 - 2 * ((y + 0.5) * inv_height)) * world.angle
    v = np.array([a, b, -1.])
    return normalize(v)

def reflect(I, N):
    # Definition from OpenGL
    return I - 2.0 * N.dot(I) * N

def refract(I, N, n1, n2):
    """
    n1 -- index of refraction of original medium
    n2 -- index of refraction of new medium
    """
    return I
    #c1 = -N.dot(I)
    #n = n1 / n2
    #c2 = m.sqrt(1 - pow(n, 2) * (1 - pow(c1, 2)))

    #return (n * I) + (n * c1 - c2) * N

def normalize(x):
    return x/np.sqrt(x.dot(x))
