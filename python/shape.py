import pyvista as pv
import numpy as np
import math, time, random, os, sys
from shape3d import cuboid_mask, elliptical_cylinder_mask

if __name__ == "__main__":
    m = np.ones((61, 124, 124))
    m[0, :, :] = 0
    
    # r = elliptical_cylinder_mask(matrix=m,
    #                 base_x=62,
    #                 base_y=62,
    #                 base_z=0,
    #                 cylinder_length=30,
    #                 radius_x=10,
    #                 radius_y=30,
    #                 yaw=0,
    #                 pitch=0,
    #                 roll=0,
    #                 taper_z=1)
    
    r = cuboid_mask(matrix=m,
                    base_x=62,
                    base_y=62,
                    base_z=0,
                    cuboid_depth=30,
                    cuboid_height=30,
                    cuboid_width=30,
                    yaw=0,
                    pitch=0,
                    roll=0,
                    taper_height=1,
                    taper_width=1)
    m[r] = 0
    grid = pv.ImageData(dimensions=np.array(m.shape) + 1)
    grid.cell_data["values"] = m.ravel(order="F")  # For correct orientation

    # Threshold to keep only m == 0
    thresholded = grid.threshold(0.1, invert=True)  # Keep values <= 0.1

    # Plot with lighting
    plotter = pv.Plotter()
    plotter.enable_shadows()
    plotter.add_mesh(thresholded, color="red", show_edges=False,
                     ambient=0.3, diffuse=0.7, specular=0.5)
    plotter.show()
