import numpy as np
import math as m

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
    n = n1/n2
    cosI = -N.dot(I)
    sinT2 = n * n * (1 - cosI * cosI)
    if sinT2 > 1:
        return None
    cosT = m.sqrt(1 - sinT2)
    return n * I + (n * cosI - cosT) * N

def normalize(x):
    return x/mag(x)

def mag(x):
    return np.sqrt(x.dot(x))
