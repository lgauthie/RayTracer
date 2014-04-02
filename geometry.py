import sys

import math as m
import numpy as np

from rayutils import normalize

class Material(object):
    """
    Class to hold information about the material
    """
    def __init__(self, color, trans, refl, shin,
                 spec=np.array([1,1,1]), diff=np.array([0.6,0.6,0.6]),
                 ref_index=1):
        self.color = color
        self.transparency = trans
        self.reflectivity = refl
        self.ref_index = ref_index
        self.shininess = shin
        self.specular = spec
        self.diffuse = diff
        self.ambient = np.array([0.2,0.2,0.2])


class Sphere(object):
    """
    Sphere class for raytracer
    """
    def __init__(self, center, radius, material):
        self.center = center
        self.radius = radius
        self.radius2 = radius ** 2
        self.material = material

    def intersected(self, raydir, rayorig):
        """
        This method is used to check if a ray intersects a Sphere instance.

        Arguments:
            raydir  -- a normalized vec3 indicating the direction of the ray.
            rayorig -- a vec3 stating the origin of the ray
        Output:
            (is_intersected, distance) :: tuple
                intersects -- is a bool to state whether the ray intersects
                distance   -- how far from the origin the intersect happens at
        """
        l = self.center - rayorig

        tca = l.dot(raydir)
        if (tca < 0):
            return (False, sys.maxint)

        d2 = l.dot(l) - tca * tca
        if (d2 > self.radius2):
            return (False, sys.maxint)

        thc = m.sqrt(self.radius2 - d2)

        t0 = tca - thc
        t1 = tca + thc
        if t0 > 0:
            t = t0
        elif t1 > 0:
            t = t1

        return (True, t)

    def get_normal(self, point):
        """
        Used to calculate the normal of a Sphere instance at a specified point

        Arguments:
            point -- a vec3, usually will be the calculated intersect with a
                     ray.
        Output:
            a normalized vec3 represeting the normal of the circle at the
            specified point.
        """
        return normalize(point - self.center)


class Plane(object):
    def __init__(self, normal, d, material):
        self.normal = normalize(normal)
        self.d = d
        self.material = material

    def intersected(self, raydir, rayorig):
        """
        This method is used to check if a ray intersects a Plane instance.

        Arguments:
            raydir  -- a normalized vec3 indicating the direction of the ray.
            rayorig -- a vec3 stating the origin of the ray
        Output:
            (is_intersected, distance) :: tuple
                intersects -- is a bool to state whether the ray intersects
                distance   -- how far from the origin the intersect happens at
        """
        den = self.normal.dot(raydir)

        # check if parallel
        if den!=0:
            nom = -(self.normal.dot(rayorig) + self.d)
            t = nom/den
            if t > 0:
                return (True, t)
        return (False, 0)

    def get_normal(self, _):
        """
        Used to calculate the normal of a Plane

        Arguments:
            _ -- used only to match the interface for Sphere, the normal of
                 a plane doesn't change along it's surface.
        Output:
            a normalized vec3 represeting the normal of plane.
        """
        return self.normal
