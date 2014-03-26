#!/usr/bin/env python
import sys

import png

import math as m
import numpy as np

class Material(object):
    def __init__(self, color, trans, refl, shin,
                 spec=np.array([1,1,1]), diff=np.array([0.6,0.6,0.6])):
        self.color = color
        self.trans = trans
        self.reflectivity = refl
        self.shininess = shin
        self.specular = spec
        self.diffuse = diff
        self.ambient = np.array([0.2,0.2,0.2])

class World(object):
    # Have Camera
    def __init__(self, width, height, fov, entities, lights):
        self.width = width
        self.height = height
        self.fov = float(fov)
        self.aspectratio = self.width/float(self.height)
        self.angle = m.tan(m.pi * 0.5 * self.fov / 180.);
        self.entities = entities
        self.max_depth = 6
        self.lights = lights

class Circle(object):
    def __init__(self, center, radius, material):
        self.center = center
        self.radius = radius
        self.radius2 = radius ** 2
        self.material = material

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

    def get_normal(self, point):
        return normalize(point - self.center)

class Plane(object):
    def __init__(self, normal, d, material):
        self.normal = normalize(normal)
        self.d = d
        self.material = material

    def intersected(self, raydir, rayorig):
        den = self.normal.dot(raydir)

        # check if parallel
        if den!=0:
            nom = -(self.normal.dot(rayorig) + self.d)
            t = nom/den
            if t > 0:
                return (True, t)
        return (False, 0)

    def get_normal(self, _):
        return self.normal

def main():
    width = 640
    height = 480

    entities = [
        Circle(np.array([0., -10003., -20.]),
               10000, Material(color=np.array([0.2, 0.5, 0.3]),
                               trans=0, refl=0.1, shin=1)),
        Circle(np.array([0., 0., -20.]),
               4, Material(color=np.array([1.00, 0.32, 0.36]),
                           trans=1, refl=0.5, shin=16)),
        Circle(np.array([5., -1., -15.]),
               4, Material(color=np.array([0.90, 0.76, 0.46]),
                           trans=1, refl=0.6, shin=32)),
        Circle(np.array([5., 0., -25.]),
               4, Material(color=np.array([0.65, 0.77, 0.97]),
                           trans=1, refl=0.2, shin=6)),
        Circle(np.array([-5.5, 0., -15.]),
               4, Material(color=np.array([0.90, 0.90, 0.90]),
                           trans=1, refl=0.1, shin=16)),
        Plane(np.array([-0.6, 0., 0.]), 3,
               Material(color=np.array([0.90, 0.20, 0.50]),
                        trans=1, refl=0.07, shin=6)),
    ]
    lights = [
        np.array([-16.001, 5.001, 15.001])
    ]

    world = World(width, height, 70., entities, lights)
    def compute_color(x, y):
        ray_dir = compute_ray_dir(x, y, world)
        return np.clip(trace(world, ray_dir) * 255, 0, 255)
    image = np.array([[compute_color(x, y) for y in range(width)]
                       for x in xrange(height)])

    image = image.reshape(height, width*3)
    with open('swatch.png', 'wb') as f:
        w = png.Writer(width, height)
        w.write(f, image)

def trace(world, ray_dir, ray_origin=np.array([0,0,0]), depth=0):
    color = np.array([0,0,0]) # Default Color to black

    # Find nearest object
    nearest_dist = sys.maxint
    nearest_entity = None
    for entity in world.entities:
        (is_intersected, dist) = entity.intersected(ray_dir, ray_origin)
        if is_intersected and dist < nearest_dist:
            nearest_entity = entity
            nearest_dist = dist

    if not nearest_entity:
        return color

    intersect = ray_origin + ray_dir * nearest_dist
    normal = nearest_entity.get_normal(intersect)

    # Shadows
    light_dir = normalize(world.lights[0] - intersect)
    inter1 = intersect + (light_dir * 0.1)
    for entity in world.entities:
        (is_intersected, dist) = entity.intersected(light_dir, inter1)
        if is_intersected:
            color = nearest_entity.material.ambient * nearest_entity.material.color
            break
    else:
        color = compute_light(nearest_entity,
                              ray_dir, normal, normalize(world.lights[0]))

    if depth <= world.max_depth and nearest_entity.material.reflectivity > 0.05:
        refl_color = trace(world, reflect(ray_dir, normal), inter1, depth+1)
        color += nearest_entity.material.reflectivity * refl_color

    return color

def compute_light(entity, ray_dir, normal, light_dir):
    reflect_dir = reflect(light_dir, normal)

    lambertian = max(light_dir.dot(normal), 0.0);
    specular = 0.0

    if lambertian > 0.0:
        spec_angle = reflect_dir.dot(ray_dir)
        specular = pow(max(spec_angle, 0.0), entity.material.shininess)

    specular_reflection = specular * entity.material.specular

    diffuse_reflection = lambertian * entity.material.diffuse

    return specular_reflection * entity.material.color \
         + diffuse_reflection * entity.material.color \
         + entity.material.ambient * entity.material.color

# make this smarter!
def compute_ray_dir(x, y, world):
    inv_width  = (1/float(world.width))
    inv_height = (1/float(world.height))
    a = (2 * ((x + 0.5) * inv_width) - 1) * world.angle * world.aspectratio
    b = (1 - 2 * ((y + 0.5) * inv_height)) * world.angle
    v = np.array([a, b, -1.])
    return normalize(v)

def reflect(I, N):
    return I - 2.0 * N.dot(I) * N

def normalize(x):
    return x/np.sqrt(x.dot(x))

if __name__=="__main__":
    main()
