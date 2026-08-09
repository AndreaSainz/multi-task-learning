import numpy as np
import tomosipo as ts


def ct_operator(H, W, angle_range, num_angles, geometry, **geometry_params):
    """
    Create a CT operator using the specified acquisition geometry.

    Parameters
    ----------
    H : int
        Height of the reconstruction volume.

    W : int
        Width of the reconstruction volume.

    angle_range : tuple of float
        Angular range of the acquisition in radians, specified as
        (start_angle, end_angle).

    num_angles : int
        Number of projection angles.

    geometry : str
        CT acquisition geometry to use. Supported geometries are:

        - "parallel" : Parallel-beam geometry.
        - "fan" : Fan-beam geometry.
        - "cone" : Cone-beam geometry.

        The value is converted to lowercase before selecting the geometry,
        so capitalization does not matter.

    **geometry_params
        Additional geometry-specific parameters required by the selected
        acquisition geometry. The required parameters depend on the value
        of geometry and are passed directly to the corresponding
        TomoSipo projection geometry constructor.

        For example, fan-beam and cone-beam geometries may require additional
        parameters such as source-detector distances or detector dimensions.

    Returns
    -------
    tomoSipo.operator
        TomoSipo CT operator corresponding to the selected acquisition
        geometry.

    Raises
    ------
    ValueError
        If geometry is not one of the supported geometries.
    """

    # Define CT volume geometry
    vg = ts.volume(shape=(1, H, W))

    # Define projection angles
    angles = np.linspace(angle_range[0], angle_range[1], num_angles, endpoint=True)    


    # Define projection geometry
    geometry = geometry.strip().lower() #lower case and without spaces

    if geometry == "parallel":
        pg = ts.parallel(angles=angles,shape=(1, H))

    elif geometry == "cone":
        pg = ts.cone(angles=angles,shape=(1, H), **geometry_params)

    elif geometry == "fan":
        pg = ts.fan(angles=angles,shape=(1, H), **geometry_params)

    else:
        raise ValueError(
            f"Unknown geometry '{geometry}'. "
            f"Supported geometries are: 'parallel', 'cone', 'fan'."
        )

    # Create CT operator
    A = ts.operator(vg, pg)

    return A