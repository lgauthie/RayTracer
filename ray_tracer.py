#!/usr/bin/env python
import sys
import png

import math as m
import numpy as np

from multiprocessing import Pool
from functools import partial

from geometry import Sphere
from geometry import Material
from geometry import Plane
from rayutils import normalize
from rayutils import reflect
from rayutils import refract
from rayutils import compute_ray_dir


class World(object):
    def __init__(self, width, height, fov, entities, lights, ms):
        self.width = width
        self.height = height
        self.fov = float(fov)
        self.aspectratio = self.width/float(self.height)
        self.angle = m.tan(m.pi * 0.5 * self.fov / 180.);
        self.entities = entities
        self.max_depth = 6
        self.lights = lights
        self.multisample = ms


def create_world(width, height):
    # TODO: Read this data in from file
    entities = [
        Sphere(np.array([0., -10003., -20.]),
               10000, Material(color=np.array([0.2, 0.5, 0.3]),
                               trans=0, refl=0.1, shin=1)),
        Sphere(np.array([0., 0., -20.]),
               4, Material(color=np.array([1.00, 0.32, 0.36]),
                           trans=0, refl=0.5, shin=16)),
        Sphere(np.array([5., -1., -15.]),
               4, Material(color=np.array([0.90, 0.76, 0.46]),
                           trans=0, refl=0.6, shin=32)),
        Sphere(np.array([5., 0., -25.]),
               4, Material(color=np.array([0.65, 0.77, 0.97]),
                           trans=0, refl=0.2, shin=6)),
        Sphere(np.array([-5.5, 0., -15.]),
               4, Material(color=np.array([0.50, 0.50, 0.50]),
                           trans=0.9, refl=0.2, shin=1)),
        Plane(np.array([-0.6, 0., 0.]), 3,
               Material(color=np.array([0.90, 0.20, 0.50]),
                        trans=0, refl=0.07, shin=6)),
    ]
    lights = [
        np.array([-16.001, 5.001, 15.001])
    ]
    return World(width, height, 70., entities, lights, ms=3)

def main():
    width = 640/2
    height = 480/2

    world = create_world(width, height)

    empty_image = [[(x, y) for y in range(width)] for x in range(height)]

    # Setup thread pools, just let python figure out the correct ammount of
    # threads needed
    p = Pool()

    # Trace our beautiful image!
    image = np.array(p.map(partial(trace_row, world), empty_image))

    # Reshape to be written to file
    image = image.reshape(height, width*3)
    with open('swatch.png', 'wb') as f:
        w = png.Writer(width, height)
        w.write(f, image)

def trace_row(world, row):
    """ Maps the function to trace a ray over the given row """
    return map(partial(compute_initial_ray, world), row)

def compute_initial_ray(world, tup):
    """ Simple wraper to start a ray from the coords held in tup """
    (x, y) = tup
    ray_dir = compute_ray_dir(x, y, world)
    color = np.array([0, 0, 0])

    ms = world.multisample
    ms_range = xrange(-(ms/2), ms/2 + 1)
    for i in ms_range:
        for j in ms_range:
            ray_dir = compute_ray_dir(x + i/float(ms), y + j/float(ms), world)
            color = color + trace(world, ray_dir)

    return np.clip((color/float(ms ** 2)) * 255, 0, 255)

def trace(world, ray_dir, ray_origin=np.array([0,0,0]), depth=0):
    color = np.array([0,0,0]) # Default Color to black

    # Find nearest object
    (nearest_entity, nearest_dist) = find_nearest(world, ray_dir, ray_origin)

    # If we don't intersect just return the background color
    if not nearest_entity:
        return color

    # Calculate the intersect point and get the objects normal at that point
    intersect = ray_origin + ray_dir * nearest_dist
    normal = nearest_entity.get_normal(intersect)

    # Do Lighting!
    for light in world.lights:
        light_dir = normalize(light - intersect)

        # Move the intersect point slightly towards to lightsoure to avoid
        # detecting collision with our current surface.
        inter1 = intersect + (light_dir * 0.1)

        # Check if there are any intersects, if so only bother with ambient
        # lighting. Otherwise carry on with the lighting calculations.
        for entity in world.entities:
            (is_intersected, dist) = entity.intersected(light_dir, inter1)
            if is_intersected:
                color = nearest_entity.material.ambient \
                      * nearest_entity.material.color
                break
        else:
            color = compute_light(nearest_entity, ray_dir, normal, normalize(light))

    # If we havent hit max recursion depth, or current material is reflective,
    # trace more rays!
    refl = nearest_entity.material.reflectivity
    trans = nearest_entity.material.transparency
    if depth <= world.max_depth:
        if refl > 0.05:
            refl_color = trace(world, reflect(ray_dir, normal), inter1, depth+1)
            color += nearest_entity.material.reflectivity * refl_color
        if trans > 0.05:
            refract_dir = refract(ray_dir, normal, 1.49, 1.3)
            trans_color = trace(world, refract_dir, intersect + (ray_dir*0.1), depth+1)
            color += nearest_entity.material.transparency * trans_color

    return color

def find_nearest(world, ray_dir, ray_origin):
    """ given the world, and a ray find the nearest intersecting object """
    nearest_dist = sys.maxint
    nearest_entity = None
    for entity in world.entities:
        (is_intersected, dist) = entity.intersected(ray_dir, ray_origin)
        if is_intersected and dist < nearest_dist:
            nearest_entity = entity
            nearest_dist = dist
    return (nearest_entity, nearest_dist)

def compute_light(entity, ray_dir, normal, light_dir):
    reflect_dir = reflect(light_dir, normal)

    lambertian = max(light_dir.dot(normal), 0.0);
    specular = 0.0

    if lambertian > 0.0:
        spec_angle = reflect_dir.dot(ray_dir)
        specular = pow(max(spec_angle, 0.0), entity.material.shininess)

    specular_reflection = specular * entity.material.specular

    diffuse_reflection = lambertian * entity.material.diffuse

    m_color = entity.material.color * (1 - entity.material.transparency)
    return specular_reflection * m_color \
         + diffuse_reflection * m_color \
         + entity.material.ambient * m_color

if __name__=="__main__":
    main()
