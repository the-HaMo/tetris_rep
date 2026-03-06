"""
Functionality of processing PolyData
"""

__author__ = "Antonio Martinez-Sanchez"

import vtk
import numpy as np
import math
from scipy import stats
from vtkmodules.util import numpy_support
from .affine import (
    angle_axis_to_quat,
    vect_to_zmat,
    rot_to_quat,
    quat_mult,
    points_distance,
)
from .tomo_utils import trilin_interp, nn_iterp


# CONSTANTS

GTRUTH_VTP_LBLS = "gt_labels"
VTK_RAY_TOLERANCE = 1e-6


def find_point_on_poly(point, poly):
    """
    Find the closest point on a poly to a reference point

    :param point: input reference point
    :param poly: poly data where the closest output point has to be found
    :return: output point
    """
    assert hasattr(point, "__len__") and (len(point) == 3)

    point_tree = vtk.vtkKdTreePointLocator()
    point_tree.SetDataSet(poly)
    point_tree.BuildLocator()
    cpoint_id = point_tree.FindClosestPoint(point)
    normals = poly.GetPointData().GetNormals()
    return np.asarray(poly.GetPoint(cpoint_id)), np.asarray(
        normals.GetTuple(cpoint_id)
    )


def gen_rand_quaternion_on_vector(vect):
    """
    Generates a unit quaternion which represents a random rotation around an input reference vector from Z-axis

    :param vect: reference vector
    :return: a quaternion which represents the rotation from Z-axis unit vector to be aligned to reference vector, plus
             a random rotation around the axis defined by the reference vector.
    """
    assert hasattr(vect, "__len__") and (len(vect) == 3)

    rnd_ang = 360.0 * np.random.random() - 180.0
    q1 = angle_axis_to_quat(rnd_ang, vect[0], vect[1], vect[2])
    M = vect_to_zmat(np.asarray(vect), mode="passive")
    q = rot_to_quat(M)
    return quat_mult(q, q1)


def gen_uni_s2_sample_on_poly(center, rad, thick, poly):
    """
    Generates a coordinate from an approximately uniformly random distribution on the intersection between a holow
    sphere a PolyData

    :param center: sphere center
    :param rad: sphere radius
    :param poly: input poly (vtkPolyData object)
    :param thick: hollow sphere thickness
    :return: the random coordinate generated or None if no intersection
    """
    assert hasattr(center, "__len__") and (len(center) == 3)
    assert (rad > 0) and (thick > 0)
    assert isinstance(poly, vtk.vtkPolyData)

    # Find poly points within rad+thick
    kdtree = vtk.vtkKdTreePointLocator()
    kdtree.SetDataSet(poly)
    kdtree.BuildLocator()
    pids = vtk.vtkIdList()
    kdtree.FindPointsWithinRadius(rad + 0.5 * thick, center, pids)

    # save_vtp(vtp_inter, './out/hold_3.vtp')

    # Get a points randomly un util a point is found in the intersection
    min_dst, n_pts = rad - 0.5 * thick, pids.GetNumberOfIds()
    for i in np.random.randint(0, n_pts, n_pts):
        pt = poly.GetPoint(pids.GetId(i))
        hold = center - pt
        if math.sqrt((hold * hold).sum()) > min_dst:
            return pt
    return None


def gen_uni_s2_sample_on_poly_inter(center, rad, poly, sph_res=360):
    """
    Generates a coordinate from an approximately uniformly random distribution on the intersection between an sphere
    a PolyData

    @Deprecated: the usage vtkIntersectionPolyDataFilter makes this function too slow

    :param center: sphere center
    :param rad: sphere radius
    :param poly: input poly (vtkPolyData object)
    :param sph_res: resolution for generating the surfaces polydata for both (longitude and latitude resolution),
                    default is 360 (1 degree resolution)
    :return: the random coordinate generated or None if no intersection
    """
    assert hasattr(center, "__len__") and (len(center) == 3)
    assert rad > 0
    assert isinstance(poly, vtk.vtkPolyData)

    # Generate the sphere polydata
    vtp_source = vtk.vtkSphereSource()
    vtp_source.SetCenter(center[0], center[1], center[2])
    vtp_source.SetRadius(rad)
    vtp_source.SetPhiResolution(36)
    vtp_source.SetThetaResolution(36)
    vtp_source.Update()
    vtp_sphere = vtp_source.GetOutput()

    # # Debug
    # from polnet.lio import save_vtp
    # save_vtp(vtp_sphere, './out/hold_1.vtp')
    # save_vtp(poly, './out/hold_2.vtp')

    # Compute polydata objects intersection
    inter_flt = vtk.vtkIntersectionPolyDataFilter()
    inter_flt.SetInputDataObject(0, vtp_sphere)
    inter_flt.SetInputDataObject(1, poly)
    inter_flt.Update()
    vtp_inter = inter_flt.GetOutput()

    # save_vtp(vtp_inter, './out/hold_3.vtp')

    # Get a point randomly on intersection
    n_pts = vtp_inter.GetNumberOfPoints()
    if n_pts == 0:
        return None
    else:
        rnd_id = np.random.randint(0, vtp_inter.GetNumberOfPoints() - 1, 1)
        return vtp_inter.GetPoint(rnd_id)


def poly_reverse_normals(poly):
    """
    Reverse the normals of an input polydata

    :param poly: input vtkPolyData object
    :return: a vtkPolyData object copy of the input but with the normals reversed
    """
    assert isinstance(poly, vtk.vtkPolyData)
    reverse = vtk.vtkReverseSense()
    reverse.SetInputData(poly)
    reverse.ReverseNormalsOn()
    reverse.Update()
    return reverse.GetOutput()


def poly_volume(poly):
    """
    Computes the volume of polydata

    :param poly: input vtkPolyData
    :return: the volume computed
    """
    assert isinstance(poly, vtk.vtkPolyData)
    mass = vtk.vtkMassProperties()
    mass.SetInputData(poly)
    return mass.GetVolume()


def poly_surface_area(poly):
    """
    Computes the surface area of polydata

    :param poly: input vtkPolyData
    :return: the volume computed
    """
    assert isinstance(poly, vtk.vtkPolyData)
    mass = vtk.vtkMassProperties()
    mass.SetInputData(poly)
    return mass.GetSurfaceArea()


def add_sfield_to_poly(
    poly, sfield, name, dtype="float", interp="NN", mode="points"
):
    """
    Add the values of a scalar field to a vtkPolyData object as point property

    :param poly: vtkPolyData objects where the scalar field values will be added
    :param sfield: input scalar field as ndarray
    :param name: string with name associated to the added property
    :param dtype: data type, valid 'float' or 'int'
    :param interp: interpolation mode, valid 'NN'-> nearest neighbour and 'trilin'-> trilinear
    :param mode: determines if the scalar field is either added to vtkPolyData points ('points', defualt) or
                 cells ('cells')
    """
    assert isinstance(sfield, np.ndarray)
    assert isinstance(name, str)
    assert (dtype == "float") or (dtype == "int")
    assert (interp == "NN") or (interp == "trilin")
    if interp == "trilin":
        interp_func = trilin_interp
    else:
        interp_func = nn_iterp
    assert (mode == "points") or (mode == "cells")

    if mode == "points":
        # Creating and adding the new property as a new array for PointData
        n_points = poly.GetNumberOfPoints()
        if dtype == "int":
            arr = vtk.vtkIntArray()
        else:
            arr = vtk.vtkFloatArray()
        arr.SetName(name)
        arr.SetNumberOfComponents(1)
        arr.SetNumberOfValues(n_points)
        for i in range(n_points):
            x, y, z = poly.GetPoint(i)
            val = interp_func(x, y, z, sfield)
            if dtype == "int":
                val = int(val)
            else:
                val = float(val)
            arr.SetValue(i, val)
        poly.GetPointData().AddArray(arr)
    else:
        # Creating and adding the new property as a new array for CellData
        if dtype == "int":
            arr = vtk.vtkIntArray()
        else:
            arr = vtk.vtkFloatArray()
        n_cells = poly.GetNumberOfCells()
        arr.SetName(name)
        arr.SetNumberOfComponents(1)
        arr.SetNumberOfValues(n_cells)
        for i in range(n_cells):
            cell = vtk.vtkGenericCell()
            poly.GetCell(i, cell)
            pts = cell.GetPoints()
            n_pts = pts.GetNumberOfPoints()
            if dtype == "int":
                values = np.zeros(shape=n_pts, dtype=int)
            else:
                values = np.zeros(shape=n_pts, dtype=float)
            for j in range(n_pts):
                x, y, z = pts.GetPoint(j)
                values[j] = interp_func(x, y, z, sfield)
            arr.SetValue(i, stats.mode(values)[0][0])
        poly.GetCellData().AddArray(arr)


def poly_mask(poly: vtk.vtkPolyData, mask: np.ndarray) -> vtk.vtkPolyData:
    """
    Removes the poly cells out of the mask

    :param poly: input poly
    :param mask: input mask
    :return: the filtered (masked) poly
    """
    assert mask.dtype == bool
    m_x, m_y, m_z = mask.shape
    del_cell_ids = vtk.vtkIdTypeArray()
    for i in range(poly.GetNumberOfCells()):
        cell = vtk.vtkGenericCell()
        poly.GetCell(i, cell)
        pts = cell.GetPoints()
        for j in range(pts.GetNumberOfPoints()):
            x, y, z = pts.GetPoint(j)
            x, y, z = int(round(x)), int(round(y)), int(round(z))
            if (
                x < 0
                or x >= m_x
                or y < 0
                or y >= m_y
                or z < 0
                or z >= m_z
                or not mask[x, y, z]
            ):
                del_cell_ids.InsertNextValue(i)
                break
    rm_filter = vtk.vtkRemovePolyData()
    rm_filter.AddInputData(poly)
    rm_filter.SetCellIds(del_cell_ids)
    rm_filter.Update()
    return rm_filter.GetOutput()


def merge_polys(poly_1, poly_2):
    """
    Merges two input poly_data in single one

    :param poly_1: input poly_data 1
    :param poly_2: input poly_data 2
    :return: an poly_data that merges the two inputs
    """
    assert isinstance(poly_1, vtk.vtkPolyData) and isinstance(
        poly_2, vtk.vtkPolyData
    )
    app_flt = vtk.vtkAppendPolyData()
    app_flt.AddInputData(poly_1)
    app_flt.AddInputData(poly_2)
    app_flt.Update()
    return app_flt.GetOutput()


def add_label_to_poly(poly, lbl, p_name, mode="cell"):
    """
    Add a label to all cells in a poly_data

    :param poly: input poly_data
    :param lbl: label (integer) value
    :param p_name: property name used for labels, if not exist in poly_dota is created
    :param mode: selected wheter the label is added to cells ('cell'), points ('point') or both ('both')
    """
    assert (mode == "cell") or (mode == "point") or (mode == "both")
    assert isinstance(poly, vtk.vtkPolyData)
    lbl, p_name = int(lbl), str(p_name)

    if mode == "cell" or mode == "both":
        arr = vtk.vtkIntArray()
        n_cells = poly.GetNumberOfCells()
        arr.SetName(p_name)
        arr.SetNumberOfComponents(1)
        arr.SetNumberOfValues(n_cells)
        for i in range(n_cells):
            arr.SetValue(i, lbl)
        poly.GetCellData().AddArray(arr)
    elif mode == "cell" or mode == "both":
        arr = vtk.vtkIntArray()
        n_points = poly.GetNumberOfPoints()
        arr.SetName(p_name)
        arr.SetNumberOfComponents(1)
        arr.SetNumberOfValues(n_points)
        for i in range(n_points):
            arr.SetValue(i, lbl)
        poly.GetPointData().AddArray(arr)


def points_to_poly_spheres(points, rad):
    """
    From an array of coordinates generates a poly_data associating a sphere centered at each point

    :param points: array or list n points with shape [n, 3]
    :param rad: sphere radius
    :return: an output poly_data
    """
    assert (
        hasattr(points, "__len__")
        and (len(points) > 0)
        and (len(points[0]) == 3)
    )
    rad = float(rad)
    app_flt = vtk.vtkAppendPolyData()

    for i in range(len(points)):
        center = points[i]
        vtp_source = vtk.vtkSphereSource()
        vtp_source.SetCenter(center[0], center[1], center[2])
        vtp_source.SetRadius(rad)
        vtp_source.Update()
        app_flt.AddInputData(vtp_source.GetOutput())

    app_flt.Update()
    return app_flt.GetOutput()


def poly_max_distance(vtp):
    """
    Computes the maximum distance in vtkPolyData

    :param vtp: input vtkPolyData
    :return: the maximum distance as real value
    """
    if vtp.GetNumberOfPoints() <= 1:
        return 0
    else:
        mx = 0
        for i in range(0, vtp.GetNumberOfPoints() - 1):
            ref_p = np.asarray(vtp.GetPoint(i))
            for j in range(i + 1, vtp.GetNumberOfPoints()):
                hold_p = np.asarray(vtp.GetPoint(j))
                hold_mx = points_distance(ref_p, hold_p)
                if hold_mx > mx:
                    mx = hold_mx
        return mx


def poly_diam(vtp):
    """
    Computes the diameter of a polydata, approximated to two times the maximumd point distance to its center of mass

    :param vtp: input vtkPolyData
    :return: the maximum distance as real value
    """
    if vtp.GetNumberOfPoints() <= 1:
        return 0
    else:
        mx = 0
        ref_p = poly_center_mass(vtp)
        for i in range(0, vtp.GetNumberOfPoints()):
            hold_p = np.asarray(vtp.GetPoint(i))
            hold_mx = points_distance(ref_p, hold_p)
            if hold_mx > mx:
                mx = hold_mx
        return mx


def poly_point_min_dst(poly, point, chull=False):
    """
    Compute the minimum distance from a point to a poly

    :param poly: input poly
    :param point: input point
    :param chull: computation mode, if True (default False) the convex hull surface is firstly extracted to avoid poly holes,
                'otherwise the minimum distance is directly computed
    :return: the minimum distance found
    """

    if poly.GetNumberOfPoints() <= 0:
        return 0
    else:
        mn = np.finfo(float).max
        if chull:
            poly = convex_hull_surface(poly)
        ref_p = np.asarray(point, dtype=float)
        for j in range(0, poly.GetNumberOfPoints()):
            hold_p = np.asarray(poly.GetPoint(j))
            hold_mn = points_distance(ref_p, hold_p)
            if hold_mn < mn:
                mn = hold_mn
        return mn


def poly_center_mass(poly):
    """
    Computes the center of mass of polydata

    :param poly: input poly
    :return: center of mass coordinates
    """
    cm_flt = vtk.vtkCenterOfMass()
    cm_flt.SetInputData(poly)
    cm_flt.Update()
    return np.asarray(cm_flt.GetCenter())


def convex_hull_surface(poly):
    """
    Extract the convex full surface of a polydata

    :param poly: input polydata
    :return: convex hull surface
    """
    convexHull = vtk.vtkDelaunay3D()
    convexHull.SetInputData(poly)
    outerSurface = vtk.vtkGeometryFilter()
    convexHull.Update()
    outerSurface.SetInputData(convexHull.GetOutput())
    outerSurface.Update()

    return outerSurface.GetOutput()


def poly_decimate(poly, dec):
    """
    Decimate a vtkPolyData

    :param poly: input vtkPolyData
    :param dec: Specify the desired reduction in the total number of polygons
               (e.g., if TargetReduction is set to 0.9,
               this filter will try to reduce the data set to 10% of its original size).
    :return: the input poly filtered
    """
    tr_dec = vtk.vtkDecimatePro()
    tr_dec.SetInputData(poly)
    tr_dec.SetTargetReduction(dec)
    tr_dec.Update()
    return tr_dec.GetOutput()


def image_to_vti(img: np.ndarray) -> vtk.vtkImageData:
    """
    Converts an image as a 2D or 3D NumPy ndarray into a VTK image
    :param img: input image as a 2D or 3D NumPy ndarray
    :return: an VTK image (vtkImageData) object
    """
    assert isinstance(img, np.ndarray)
    n_D = len(img.shape)
    assert n_D == 2 or n_D == 3
    data_type = vtk.VTK_FLOAT
    shape = img.shape

    flat_data_array = img.flatten()
    vtk_data = numpy_support.numpy_to_vtk(
        num_array=flat_data_array, deep=True, array_type=data_type
    )

    img = vtk.vtkImageData()
    img.GetPointData().SetScalars(vtk_data)
    if n_D == 2:
        img.SetDimensions(shape[0], shape[1], 1)
    else:
        img.SetDimensions(shape[0], shape[1], shape[2])

    return img


def save_vti(img: vtk.vtkImageData, fname: str):
    """
    Stores a VTK image
    :param img: the input image as a VTK image
    :param fname: file name and path ended with .vtk or .vti
    :return:
    """
    assert isinstance(img, vtk.vtkImageData)
    assert isinstance(fname, str)
    assert fname.endswith(".vtk") or fname.endswith(".vti")

    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(fname)
    writer.SetInputData(img)
    if writer.Write() != 1:
        print("ERROR: unknown error writting the .vti file!!!")


def iso_surface(tomo, th, flp=None, closed=False, normals=None):
    """
    Iso-surface on an input 3D volume

    :param tomo: input 3D numpy array
    :param th: iso-surface threshold
    :param flp: if not None (default) it specifies the axis to flip (valid: 0, 1 or 3)
    :param closed: if True (default False) if forces to generate a closed surface, VERY IMPORTANT: closed output
    is only guaranteed for input boolean tomograms
    :param normals: normals orientation, valid None (default), 'inwards' and 'outwards'. Any value different from None
                    reduces surface precision
    :return: a vtkPolyData object only made up of triangles
    """

    # Marching cubes configuration
    march = vtk.vtkMarchingCubes()
    tomo_vtk = numpy_to_vti(tomo)
    if closed:
        # print str(tomo_vtk.GetExtent()), str(tomo.shape)
        padder = vtk.vtkImageConstantPad()
        padder.SetInputData(tomo_vtk)
        padder.SetConstant(0)
        padder.SetOutputWholeExtent(
            -1, tomo.shape[0], -1, tomo.shape[1], -1, tomo.shape[2]
        )
        padder.Update()
        tomo_vtk = padder.GetOutput()

    # Flipping
    if flp is not None:
        flp_i = int(flp)
        if (flp_i >= 0) and (flp_i <= 3):
            fliper = vtk.vtkImageFlip()
            fliper.SetFilteredAxis(flp_i)
            fliper.SetInputData(tomo_vtk)
            fliper.Update()
            tomo_vtk = fliper.GetOutput()

    # Running Marching Cubes
    march.SetInputData(tomo_vtk)
    march.SetValue(0, th)
    march.Update()
    hold_poly = march.GetOutput()

    # Filtering
    hold_poly = poly_filter_triangles(hold_poly)

    # Normals orientation
    if normals is not None:
        orienter = vtk.vtkPolyDataNormals()
        orienter.SetInputData(hold_poly)
        orienter.AutoOrientNormalsOn()
        if normals == "inwards":
            orienter.FlipNormalsOn()
        orienter.Update()
        hold_poly = orienter.GetOutput()

    if closed and (not is_closed_surface(hold_poly)):
        raise RuntimeError

    return hold_poly


def is_closed_surface(poly):
    """
    Checks if an input vtkPolyData is a closed surface

    :param poly: input vtkPolyData to check
    :return: True is the surface is closed, otherwise False
    """
    selector = vtk.vtkSelectEnclosedPoints()
    selector.CheckSurfaceOn()
    selector.SetSurfaceData(poly)
    if selector.GetCheckSurface() > 0:
        return True
    else:
        return False


def poly_filter_triangles(poly):
    """
    Filter a vtkPolyData to keep just the polys which are triangles

    :param poly: input vtkPolyData
    :return: a copy of the input poly but filtered
    """
    cut_tr = vtk.vtkTriangleFilter()
    cut_tr.SetInputData(poly)
    cut_tr.PassVertsOff()
    cut_tr.PassLinesOff()
    cut_tr.Update()
    return cut_tr.GetOutput()


def numpy_to_vti(array, spacing=[1, 1, 1]):
    """
    Converts a 3D numpy array into vtkImageData object

    :param array: 3D numpy array
    :param spacing: distance between pixels
    :return: a vtkImageData object
    """

    # Flattern the input array
    array_1d = numpy_support.numpy_to_vtk(
        num_array=np.reshape(array, -1, order="F"),
        deep=True,
        array_type=vtk.VTK_FLOAT,
    )

    # Create the new vtkImageData
    nx, ny, nz = array.shape
    image = vtk.vtkImageData()
    image.SetSpacing(spacing)
    image.SetDimensions(nx, ny, nz)
    image.AllocateScalars(vtk.VTK_FLOAT, 1)
    image.GetPointData().SetScalars(array_1d)

    return image


def poly_threshold(poly, p_name, mode="points", low_th=None, hi_th=None):
    """
    Threshold a vtkPolyData according the values of a property

    :param poly: vtkPolyData to threshold
    :param p_name: property name for points
    :param mode: determines if the property is associated to points data 'points' (default) or 'cells'
    :low_th: low threshold value, default None then the minimum property value is assigned
    :hi_th: high threshold value, default None then the maximum property value is assigned
    :return: the threshold vtkPolyData
    """

    # Input parsing
    prop = None
    assert (mode == "points") or (mode == "cells")
    if mode == "points":
        n_arrays = poly.GetPointData().GetNumberOfArrays()
        for i in range(n_arrays):
            if p_name == poly.GetPointData().GetArrayName(i):
                prop = poly.GetPointData().GetArray(p_name)
                break
    else:
        n_arrays = poly.GetCellData().GetNumberOfArrays()
        for i in range(n_arrays):
            if p_name == poly.GetCellData().GetArrayName(i):
                prop = poly.GetCellData().GetArray(p_name)
                break
    assert prop is not None
    if (low_th is None) or (hi_th is None):
        rg_low, rg_hi = prop.GetRange()
    if low_th is None:
        low_th = rg_low
    if hi_th is None:
        hi_th = rg_hi

    # Points thresholding filter
    th_flt = vtk.vtkThreshold()
    th_flt.SetInputData(poly)
    if mode == "cells":
        th_flt.SetInputArrayToProcess(
            0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_CELLS, p_name
        )
    else:
        th_flt.SetInputArrayToProcess(
            0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, p_name
        )
    # th_flt.ThresholdByUpper(.5)
    # th_flt.ThresholdBetween(low_th, hi_th)
    th_flt.SetLowerThreshold(low_th)
    th_flt.SetUpperThreshold(hi_th)
    th_flt.AllScalarsOff()
    th_flt.Update()

    surf_flt = vtk.vtkDataSetSurfaceFilter()
    surf_flt.SetInputData(th_flt.GetOutput())
    surf_flt.Update()

    return surf_flt.GetOutput()


def poly_scale(in_vtp, s):
    """
    Applies scaling transformation to a vtkPolyData

    :param in_vtp: input vtkPolyData
    :param s: scaling factor
    :return: the transformed vtkPolyData
    """
    # Translation
    box_sc = vtk.vtkTransform()
    box_sc.Scale(s, s, s)
    tr_box = vtk.vtkTransformPolyDataFilter()
    tr_box.SetInputData(in_vtp)
    tr_box.SetTransform(box_sc)
    tr_box.Update()
    return tr_box.GetOutput()
