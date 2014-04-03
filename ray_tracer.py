#!/usr/bin/env python
import sys
import png
import argparse

import math as m
import numpy as np

from multiprocessing import Pool
from functools import partial
from random import uniform

from geometry import Sphere
from geometry import Material
from geometry import Plane
from rayutils import normalize
from rayutils import reflect
from rayutils import refract
from rayutils import compute_ray_dir
from rayutils import mag


class World(object):
    def __init__(self, width, height, fov, entities, lights, ms, randsample=False):
        self.width = width
        self.height = height
        self.fov = float(fov)
        self.aspectratio = self.width/float(self.height)
        self.angle = m.tan(m.pi * 0.5 * self.fov / 180.);
        self.entities = entities
        self.max_depth = 5
        self.lights = lights
        self.multisample = ms
        self.randsample = randsample


def create_world1(width, height):
    entities = [
        Sphere(np.array([0., -10003., -30.]),
               10000, Material(color=np.array([0.2, 0.5, 0.3]),
                               trans=0, refl=0.1, shin=1)),
        Sphere(np.array([0., 0., -30.]),
               4, Material(color=np.array([1.00, 0.32, 0.36]),
                           trans=0, refl=0.5, shin=16)),
        Sphere(np.array([5., -1., -25.]),
               4, Material(color=np.array([0.90, 0.76, 0.46]),
                           trans=0, refl=0.6, shin=32)),
        Sphere(np.array([5., 2., -35.]),
               4, Material(color=np.array([0.65, 0.77, 0.97]),
                           trans=0, refl=0.2, shin=6)),
        Sphere(np.array([-4.5, 0., -25.]),
               4, Material(color=np.array([0.50, 0.50, 0.50]),
                           trans=0.8, refl=0.2, shin=1, ref_index=1.00332)),
        Plane(np.array([-0.6, 0., 0.]), 3,
               Material(color=np.array([0.90, 0.20, 0.50]),
                        trans=0, refl=0.07, shin=6)),
    ]
    lights = [
        np.array([-16.001, 5.001, 15.001])
    ]
    return World(width, height, 45., entities, lights, ms=1)

def create_world2(width, height):
    entities = [
        Sphere(np.array([-2.5, 4., -25.]),
               2, Material(color=np.array([1.00, 0.20, 0.30]),
                           trans=0.1, refl=0.4, shin=1, ref_index=1.5)),
        Sphere(np.array([-2.5, 2., -25.]),
               2, Material(color=np.array([1.00, 0.20, 0.60]),
                           trans=0.1, refl=0.4, shin=1, ref_index=1.5)),
        Sphere(np.array([-4.5, 0., -25.]),
               2, Material(color=np.array([1.00, 0.20, 0.70]),
                           trans=0.2, refl=0.4, shin=1, ref_index=1.0)),
        Sphere(np.array([-3.5, -1., -20.]),
               4, Material(color=np.array([0.00, 0.70, 0.00]),
                           trans=0.95, refl=0.2, shin=1, ref_index=1.332)),
        Plane(np.array([-1.0, 0.0, 0.1]), 5,
               Material(color=np.array([0.00, 0.00, 0.70]),
                        trans=0, refl=0.1, shin=6)),
    ]
    lights = [
        np.array([-10.001, 5.001, 5.001])
    ]
    return World(width, height, 45., entities, lights, ms=1)

def create_world3(width, height):
    entities = [
        Sphere(np.array([-2.5, 4., -25.]),
               2, Material(color=np.array([0.00, 0.00, 1.00]),
                           trans=0.1, refl=0.7, shin=16, ref_index=1.5)),
        Sphere(np.array([-6.5, 6., -25.]),
               2, Material(color=np.array([0.00, 1.00, 0.00]),
                           trans=0.1, refl=0.7, shin=14, ref_index=1.5)),
        Sphere(np.array([0.5, 0., -25.]),
               2, Material(color=np.array([1.00, 0.00, 0.00]),
                           trans=0.2, refl=0.7, shin=1, ref_index=1.0)),
        Sphere(np.array([-6.5, -4., -25.]),
               2, Material(color=np.array([0.00, 0.00, 1.00]),
                           trans=0.1, refl=0.7, shin=16, ref_index=1.5)),
        Sphere(np.array([-2.5, -6., -25.]),
               2, Material(color=np.array([0.00, 1.00, 0.00]),
                           trans=0.1, refl=0.7, shin=14, ref_index=1.5)),
        Sphere(np.array([-8.5, 0., -25.]),
               2, Material(color=np.array([1.00, 0.00, 0.00]),
                           trans=0.2, refl=0.7, shin=1, ref_index=1.0)),
        Plane(np.array([-2.0, 0.0, 0.1]), 5,
               Material(color=np.array([0.60, 0.60, 0.60]),
                        trans=0, refl=1.0, shin=6)),
    ]
    lights = [
        np.array([-15.001, 5.001, 5.001])
    ]
    return World(width, height, 75., entities, lights, ms=1)

def main():
    width = 640/3
    height = 480/3

    p = argparse.ArgumentParser(description="parse some things.")
    p.add_argument("--height")
    p.add_argument("--width")
    p.add_argument("--world")
    p.add_argument("--ms")
    p.add_argument("--depth")
    p.add_argument("--rsample")
    opts = vars(p.parse_args())

    if opts["height"]:
        height = int(opts["height"])
    if opts["width"]:
        width = int(opts["width"])

    world = create_world1(width, height)
    if opts["world"] == "1":
        world = create_world1(width, height)
    elif opts["world"] == "2":
        world = create_world2(width, height)
    elif opts["world"] == "3":
        world = create_world3(width, height)

    if opts["ms"]:
        world.multisample = int(opts["ms"])

    if opts["depth"]:
        world.max_depth = int(opts["depth"])

    if opts["rsample"]:
        world.randsample = True

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
    if world.randsample:
        for _ in range(ms):
            ray_dir = compute_ray_dir(x + uniform(-0.5, 0.5),
                                      y + uniform(-0.5, 0.5),
                                      world)
            color = color + trace(world, ray_dir)
        return np.clip((color/float(ms)) * 255, 0, 255)
    else:
        for xx in range(ms):
            for yy in range(ms):
                ray_dir = compute_ray_dir(x + (xx - ((ms-1)/2.))/float(ms),
                                          y + (yy - ((ms-1)/2.))/float(ms),
                                          world)
                color = color + trace(world, ray_dir)
        return np.clip((color/float(ms * ms)) * 255, 0, 255)

def trace(world, ray_dir, ray_origin=np.array([0,0,0]), depth=0, cur_ref_index=1):
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
        light_dist = mag(light - intersect)

        # Move the intersect point slightly towards to lightsoure to avoid
        # detecting collision with our current surface.
        inter1 = intersect + (light_dir * 0.1)

        # Check if there are any intersects, if so only bother with ambient
        # lighting. Otherwise carry on with the lighting calculations.
        for entity in world.entities:
            (is_intersected, dist) = entity.intersected(light_dir, inter1)
            if is_intersected:
                #color = nearest_entity.material.ambient \
                #      * nearest_entity.material.color
                color = compute_light(nearest_entity, ray_dir, normal, normalize(light), light_dist) \
                      * max((entity.material.transparency), 0.3)
                break
        else:
            color = compute_light(nearest_entity, ray_dir, normal, normalize(light), light_dist)

    # If we havent hit max recursion depth, or current material is reflective,
    # trace more rays!
    refl = nearest_entity.material.reflectivity
    trans = nearest_entity.material.transparency
    if depth <= world.max_depth:
        if refl > 0.05:
            refl_color = trace(world, reflect(ray_dir, normal), inter1, depth+1, cur_ref_index=cur_ref_index)
            color += nearest_entity.material.reflectivity * refl_color
        if trans > 0.05:
            if nearest_entity.material.ref_index == cur_ref_index:
                new_ref_index = 1
            else:
                new_ref_index = nearest_entity.material.ref_index
            refract_dir = refract(ray_dir, normal, cur_ref_index, new_ref_index)
            if refract_dir != None:
                trans_color = trace(world, refract_dir, intersect + (ray_dir*0.1), depth+1, cur_ref_index=new_ref_index)
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

def compute_light(entity, ray_dir, normal, light_dir, dist):
    reflect_dir = reflect(light_dir, normal)

    lambertian = max(light_dir.dot(normal), 0.0);
    specular = 0.0
    falloff = 2.0

    if lambertian > 0.0:
        spec_angle = reflect_dir.dot(ray_dir)
        specular = pow(max(spec_angle, 0.0), entity.material.shininess)

    specular_reflection = specular * entity.material.specular

    diffuse_reflection = lambertian * entity.material.diffuse

    m_color = entity.material.color * (1 - entity.material.transparency)
    falloff_amount = (pow(dist, -falloff) if dist != 0 else 1) * 1000
    return specular_reflection * m_color * falloff_amount \
         + diffuse_reflection * m_color * falloff_amount \
         + entity.material.ambient * m_color

if __name__=="__main__":
    main()
