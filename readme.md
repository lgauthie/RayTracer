Python Raytracer!
=================

By: Lee D. Gauthier
UVic ID: V00681729

How To Run
----------

`
python ray_tracer.py --world 3 --ms 3 --width 640 --height 480
`

- --world: The scene to render. Either 1 2 or 3
- --ms: The number to multisample. Creates an ms by ms grid for sampling.
- --width: The scenes width in pixels
- --height: The scenes height in pixels
- --depth: How deep each ray is allowed to recurse
- --rsample: Use random sampling instead of a grid

Sources
-------
- [Refraction found here](http://graphics.stanford.edu/courses/cs148-10-summer/docs/2006--degreve--reflection\_refraction.pdf)
- [Scratch pixel for sphere collision](http://www.scratchapixel.com/lessons/3d-basic-lessons/lesson-1-writing-a-simple-raytracer/)

Marking Rubrick
---------------

* (Done) 1 pt for code comments & README.txt file
* (Done) 2 pt for writing an image to disc
* (Done) 2 pt for generating the rays through each pixel and intersecting with a plane as floor
* (Done) Select one of these options:
   - (Done) 4 pt for intersecting rays with a sphere
   - 4 pt for using mesh models
* (Done) 2 pt for calculating the diffuse and specular color
* (Done) 1 pt for calculating if a surface point is in the shadow or not
* (Done) 2 pt for calculating reflections with arbitrary recursion depth
* (Done) 1 pt for shooting an arbitrary number of rays per pixel (sampling)
* (Done) 3 pt for combining Phong illumination with shadowing and reflections

18 TOTAL

Bonus:

- (Done) 0.5 pt for adding refractions

