#!/usr/bin/env python
import sys

import png

import math as m
import numpy as np

class World(object):
    def __init__(self, width, height, fov, entities):
        self.width = width
        self.height = height
        self.fov = float(fov)
        self.aspectratio = self.width/float(self.height)
        self.angle = m.tan(m.pi * 0.5 * self.fov / 180.);
        self.entities = entities
        self.max_depth = 5

class Circle(object):
    def __init__(self, center, radius, color, trans, refl):
        self.center = center
        self.radius = radius
        self.radius2 = radius ** 2
        self.color = color
        self.trans = trans
        self.refl = refl

    def intersected(self, raydir, rayorig):
        l = self.center - rayorig

        tca = l.dot(raydir)
        if (tca < 0):
            return (False, sys.maxint)

        d2 = l.dot(l) - tca * tca;
        if (d2 > self.radius2):
            return (False, sys.maxint)

        thc = m.sqrt(self.radius2 - d2)

        t = tca - thc
        return (True, t)

class Plane(object):
    def __init__(self, normal, d):
        self.normal = normal
        self.d = d
        self.color = np.array([1.,0.4,1.])

    def intersected(self, raydir, rayorig):
        den = self.normal.dot(raydir)

        # check if parallel
        if den!=0:
            nom = -(self.normal.dot(rayorig) + self.d)
            t = nom/den
            if t > 0:
                return (True, t)
        return (False, 0)

def main():
    width = 300
    height = 200

    entities = [
        Circle(np.array([0., -10004., -20.]),
               10000, np.array([0.2, 0.2, 0.2]), 0, 0.0),
        Circle(np.array([0., 0., -20.]),
               4, np.array([1.00, 0.32, 0.36]), 1, 0.5),
        Circle(np.array([5., -1., -15.]),
               4, np.array([0.90, 0.76, 0.46]), 1, 0),
        Circle(np.array([5., 0., -25.]),
               3, np.array([0.65, 0.77, 0.97]), 1, 0.0),
        Circle(np.array([-5.5, 0, -15]),
               3, np.array([0.90, 0.90, 0.90]), 1, 0.0),
        Plane(np.array([-1., 0., 0.]), 3),
    ]

    world = World(width, height, 60., entities)

    image = np.array([[trace(x, y, world) for y in range(width)]
                       for x in xrange(height)])

    image = image.reshape(height, width*3)
    with open('swatch.png', 'wb') as f:
        w = png.Writer(width, height)
        w.write(f, image)

def trace(x, y, world, ray_origin=np.array([0,0,0]), depth=0):
    color = np.array([0,0,0])
    ray_dir = compute_ray_dir(x, y, world)
    smallest_dist = sys.maxint
    for entity in world.entities:
        (is_intersected, dist) = entity.intersected(ray_dir, ray_origin)
        if is_intersected and dist < smallest_dist:
            color = entity.color
            smallest_dist = dist

    #if True: # Ray hits more than one boundry box
    #    # Find the boundary box near to the camera position
    #    # so it will find the object near the eye.
    #    # Get the object inside the nearest boundary
    #    pass
    #    if True: # Ray hits the object
    #        # Get intersection points
    #        pass
    #        if True: # More than one intersection
    #            # Find nearest to camera
    #            pass
    #        else:
    #            # you have two options (choose one):
    #            #     1: don't color the pixel, the Color remains black
    #            #     2: The point is our interested point;
    #            pass
    #    # now give the intersected point a proper color
    #    color = np.array([255,0,0]) # Calculate Color
    return color * 255


def compute_ray_dir(x, y, world):
    inv_width = (1/float(world.width))
    inv_height = (1/float(world.height))
    a = (2 * ((x + 0.5) * inv_width) - 1) * world.angle * world.aspectratio
    b = (1 - 2 * ((y + 0.5) * inv_height)) * world.angle
    # a = m.tan(world.fov/2.)*((x-(world.width/2.))/world.width/2.)
    # b = m.tan(world.fov/2.)*(((world.height/2.)-y)/world.height/2.)
    v = np.array([a, b, -1.])
    return normalize(v)
    #return world.eye + (v/ np.sqrt(v.dot(v)))

class Camera(object):
    def __init__(self, eye, fovx, fovy, width, height):
        self.fovx = fovx
        self.fovy = fovy
        self.width = width
        self.height = height

def normalize(x):
    return x/np.sqrt(x.dot(x))

def look_at(eye, at, up):
    mat = np.identity(4)

    e = eye
    a = e - at
    b = up

    w = normalize(a)
    u = normalize(np.cross(b, w))
    v = np.cross(w, u)

    mat[0,0] = u[0]
    mat[1,0] = u[1]
    mat[2,0] = u[2]

    mat[0,1] = v[0]
    mat[1,1] = v[1]
    mat[2,1] = v[2]

    mat[0,2] = w[0]
    mat[1,2] = w[1]
    mat[2,2] = w[2]

    # Set up translation
    mat[0,3] =  -(u[0] * e[0]) - (u[1] * e[1]) - (u[2] * e[2]);
    mat[1,3] =  -(v[0] * e[0]) - (v[1] * e[1]) - (v[2] * e[2]);
    mat[2,3] =  -(w[0] * e[0]) - (w[1] * e[1]) - (w[2] * e[2]);
    print mat

if __name__=="__main__":
    main()
