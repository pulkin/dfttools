"""
This submodule contains data visualization routines.
"""
import base64
import math
from xml.etree import ElementTree
from itertools import product
from tempfile import NamedTemporaryFile

from .types import Basis, Cell, Grid
from .data import element_number, element_size, element_color_convention

import numpy
import numericalunits

try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO


__xyz2i__ = {'x': 0, 'y': 1, 'z': 2, 0: 0, 1: 1, 2: 2}


def __fadeout_z__(color, z, mx, mn, strength, bg):
    alpha = min(max((z - mn) / (mx - mn) * strength, 0), 1)
    return (numpy.array(color, dtype=numpy.float64) * (1 - alpha) + numpy.array(bg,
                                                                                dtype=numpy.float64) * alpha).astype(
        numpy.int64)


def __dark__(color, delta=0.4):
    return (numpy.array(color, dtype=numpy.float64) * (1 - delta)).astype(numpy.int64)


def __light__(color, delta=0.4):
    return (255 - (255 - numpy.array(color, dtype=numpy.float64)) * (1 - delta)).astype(numpy.int64)


def __svg_color__(color):
    return "rgb({:.0f},{:.0f},{:.0f})".format(*color)


def __window__(p1, p2, window):
    inside = lambda x, y: (x > window[0]) and (y > window[1]) and (x < window[2]) and (y < window[3])

    if not inside(*p1):
        p1, p2 = p2, p1

    if not inside(*p1):
        return None, None

    if inside(*p2):
        return p1, p2

    if p2[0] < window[0] or p2[0] > window[2]:
        k = (p2[1] - p1[1]) / (p2[0] - p1[0])
        b = p2[1] - k * p2[0]
        p2[0] = window[0] if p2[0] < window[0] else window[2]
        p2[1] = k * p2[0] + b

    if p2[1] < window[1] or p2[1] > window[3]:
        k = (p2[0] - p1[0]) / (p2[1] - p1[1])
        b = p2[0] - k * p2[1]
        p2[1] = window[1] if p2[1] < window[1] else window[3]
        p2[0] = k * p2[1] + b

    return p1, p2


def svgwrite_unit_cell(
        cell,
        svg,
        camera=None,
        camera_top=None,
        insert=(0, 0),
        size=(600, 600),
        circle_size=0.4,
        bond_size=1,
        circle_opacity=None,
        margin=6,
        show_cell=False,
        show_atoms=True,
        show_bonds=True,
        show_legend=True,
        show_numbers=False,
        show_vectors=False,
        vectors_offset=None,
        fadeout_strength=0.8,
        bg=(0xFF, 0xFF, 0xFF),
        bond_ratio=1,
        hook_atomic_color=None,
        coordinates='right',
        invisible=None,
        title=None,
        font_family=None,
        font_size=None,
        font_size_small=None,
        return_coords=False,
        overlay_opacity=0.8,
        perspective_correction=0,
        circle_stroke=0.1,
):
    """
    Creates an svg drawing of a unit cell.

    Args:

        cell (Cell): the cell to be visualized;

        svg (str, svgwrite.Drawing): either file name to save the drawing
        to or an ``svgwrite.Drawing`` object to draw with.

    Kwargs:

        camera (str, array): the direction of a camera: either 'x','y' or
        'z' or an arbitrary 3D vector;

        camera_top (array): a vector pointing up;

        insert (array): a top-left corner of the drawing;

        size (array): size of the bounding box;

        circle_size (float): relative radius of atoms;

        bond_size (float): relative radius of bonds;

        circle_opacity (float,array): opacity of circles;

        margin (float): size of the margin in all directions;

        show_cell (bool, str): if True draws the unit cell edges projected,
        if 'invisible' the unit cell is invisible;

        show_atoms (bool): if True draws atoms;

        show_bonds (bool): if True draws bonds;

        show_legend (bool): if True draws legend;

        show_numbers (bool): if True shows numbers corresponding to the
        atomic order in the unit cell;

        show_vectors (bool, tuple): if True, shows unit vectors in the bottom-left
        corner;

        vectors_offset (float, tuple): sets a custom offset for vectors (in px);

        fadeout_strength (float): amount of fadeout applied to more distant atoms;

        bg (array): an integer array defining background color;

        bond_ratio (float): scale factor to determine whether the bond
        is rendered;

        coordinates (str): the coordinate system, either 'left' or 'right';

        hook_atomic_color (function): a function accepting integer (atom
        ID) and a 3-element list (suggested RGB color) and returning a
        new color of the atom;

        invisible (str,array): make specified atoms invisible. If 'auto'
        specified, creates a supercell and makes all cell replica
        invisible. The bonds of invisible atoms will still be present on
        the final image;

        title (str): a title to the drawing presented in the top left
        corner;

        font_family (dict): the font family for captions;

        font_size (float): the size of the font in points;

        font_size_small (float): the size of the small font in points;

        return_coords (bool): if True, additionally returns a 2-tuple of
        the svg group containing coordinates as well as an array with
        all atomic coordinates inside the group;

        overlay_opacity (float): the opacity of overlays;

        perspective_correction (float): the degree of perspective correction:
        from 0 (no correction) to infinity (everything is projected into a
        point);

        circle_stroke (float): the size of the stroke of circles in units
        of radius;

    Returns:

        An ```svgwrite.Drawing`` object. The object is saved if it was
        created inside this method.
    """
    if font_family is None:
        font_family = "Liberation Sans"
    if font_size is None:
        font_size = 10
    if font_size_small is None:
        font_size_small = 0.8 * font_size

    font_props = dict(font_family=font_family, font_size="{:.1f}pt".format(font_size))
    font_props_small = dict(font_family=font_family, font_size="{:.1f}pt".format(font_size_small))

    if invisible is None:
        visible = numpy.ones(cell.size, dtype=bool)

    elif invisible == 'auto':
        N = cell.size
        initial_cell = cell
        cell = cell.repeated(3, 3, 3)
        visible = numpy.array([False] * 13 * N + [True] * N + [False] * 13 * N, dtype=bool)

    else:
        visible = numpy.logical_not(invisible)

    insert = numpy.array(insert, dtype=numpy.float64)
    size = numpy.array(size, dtype=numpy.float64)
    osize = size - 2 * margin

    if isinstance(svg, str):
        import svgwrite
        save = True
        svg = svgwrite.Drawing(svg, size=(size).tolist(), profile='full')
    else:
        save = False

    # Camera vector
    if camera is None:
        # Determine the largest face
        cv = cell.vectors
        if not show_cell:
            c = cell.coordinates
            cmin = c.min(axis=0)
            cmax = c.max(axis=0)
            cv = cv * (cmax-cmin)[:, numpy.newaxis]
        areas = list((numpy.cross(cv[(i + 1) % 3], cv[(i + 2) % 3]) ** 2).sum() for i in range(3))
        camera = "xyz"[numpy.argmax(areas)]

    try:
        camera = {
            "x": (-1, 0, 0),
            "y": (0, -1, 0),
            "z": (0, 0, -1),
        }[camera]
    except (KeyError, TypeError):
        pass
    camera = numpy.array(camera, dtype=numpy.float64)

    if len(camera) == 2:
        theta, phi = camera * numpy.pi
        camera = numpy.array((
            numpy.sin(theta) * numpy.cos(phi),
            numpy.sin(theta) * numpy.sin(phi),
            numpy.cos(theta),
        ))
    camera_z = camera / numpy.linalg.norm(camera)

    # Camera top vector
    if camera_top is None:
        # Determine lattice vector with the longest projection
        proj = ((cell.vectors - numpy.dot(cell.vectors, camera)[:, numpy.newaxis] * camera[numpy.newaxis, :]) ** 2).sum(
            axis=-1)
        camera_top = numpy.cross(camera, cell.vectors[numpy.argmax(proj)])
    else:
        camera_top = numpy.array(camera_top, dtype=numpy.float64)

    camera_x = numpy.cross(camera, camera_top)
    camera_y = camera_top - numpy.dot(camera_top, camera_z) * camera_z

    if numpy.linalg.norm(camera_x) == 0 or numpy.linalg.norm(camera_y) == 0:
        raise ValueError("The 'camera' and 'camera_top' vectors cannot be collinear")

    camera_x /= numpy.linalg.norm(camera_x)
    camera_y /= numpy.linalg.norm(camera_y)

    # Calculate projection matrix
    projection = Basis((
        camera_x,
        camera_y,
        camera_z,
    ))

    # Project atomic coordinates onto the plane
    projected = projection.transform_from(cell, cell.coordinates)

    # Collect elements
    e_color = tuple(element_color_convention[i.lower()] for i in cell.values)
    e_sizes = numpy.array(tuple(element_size[i.lower()] for i in cell.values)) * numericalunits.angstrom
    e_size = e_sizes[:, 0]
    e_covsize = e_sizes[:, 1]
    e_covsize *= bond_ratio

    # Determine boundaries
    b_min = numpy.min((projected - e_size[..., numpy.newaxis] * circle_size)[visible, :], axis=0)
    b_max = numpy.max((projected + e_size[..., numpy.newaxis] * circle_size)[visible, :], axis=0)

    if show_cell:

        # Project unit cell edges ...
        if isinstance(invisible, str) and invisible == 'auto':
            projected_edges = projection.transform_from_cartesian(
                initial_cell.edges + initial_cell.vectors.sum(axis=0)[numpy.newaxis, :])
        else:
            projected_edges = projection.transform_from_cartesian(cell.edges)

        # ... and modify boundaries
        b_min = numpy.minimum(b_min, projected_edges.reshape(-1, projected_edges.shape[-1]).min(axis=0))
        b_max = numpy.maximum(b_max, projected_edges.reshape(-1, projected_edges.shape[-1]).max(axis=0))

    center = 0.5 * (b_min + b_max)[:2]
    scale = (osize / (b_max[:2] - b_min[:2])).min()
    shift = 0.5 * osize - center * scale
    shift = numpy.append(shift, 0)

    projected *= scale
    projected += shift[numpy.newaxis, :]

    if show_cell:
        projected_edges *= scale
        projected_edges += shift[numpy.newaxis, :]

    b_max *= scale
    b_min *= scale
    center *= scale
    b_max += shift
    b_min += shift
    center += shift[:2]

    center = center

    # Perspective correction
    def correct_perspective(a, z, center=None):
        if center is not None:
            center = center.reshape((1,) * (a.ndim-1) + (a.shape[-1],))
        z = z.copy()
        z -= b_min[2]
        z /= (b_max[2] - b_min[2])
        alpha = 1./ (z * perspective_correction + 1)
        pshape = (1,) * (a.ndim - 1) + (2,)
        if center is not None:
            a -= center
            a *= alpha[..., numpy.newaxis]
            a += center
        else:
            a *= alpha
        return a

    projected[..., :2] = correct_perspective(projected[..., :2], projected[..., 2], center=center)
    e_size = correct_perspective(e_size, projected[..., 2])

    if show_cell:
        projected_edges[..., :2] = correct_perspective(projected_edges[..., :2], projected_edges[..., 2], center=center)

    # Calculate base colors
    colors_base = tuple(__fadeout_z__(e_color[i], projected[i, 2], b_max[2], b_min[2], fadeout_strength, bg if bg is not None else (0xFF, 0xFF, 0xFF)) for i in
                        range(cell.size))
    if hook_atomic_color:
        if invisible != "auto":
            colors_base = tuple(hook_atomic_color(i, c) for i, c in enumerate(colors_base))
        else:
            colors_base = colors_base[:13 * N] + tuple(
                hook_atomic_color(i, c) for i, c in enumerate(colors_base[13 * N:14 * N])) + colors_base[14 * N:]

    # Arrays for storing objects with z-index
    obj = []
    obj_z = []

    # Group holding the image
    group = svg.g()
    group.translate(*tuple(insert))
    svg.add(group)

    # BG
    if bg is not None:
        group.add(svg.rect(
            insert=(0, 0),
            size=size,
            fill=__svg_color__(bg),
        ))

    # Subgroup with atoms etc
    subgroup = svg.g()
    group.add(subgroup)

    if coordinates == 'left':
        subgroup.translate(margin, margin)

    elif coordinates == 'right':
        subgroup.scale(1.0, -1.0)
        subgroup.translate(margin, -size[1] + margin)

    else:
        raise ValueError("Parameter 'coordinates' should be either 'left' or 'right'")

    if show_cell == True:

        # Draw unit cell edges
        for pair in projected_edges:
            obj.append(svg.line(
                start=pair[0][:2],
                end=pair[1][:2],
                stroke="black",
                opacity=0.1,
                stroke_width=0.01 * max(*size),
            ))
            obj_z.append(0.5 * (pair[0, 2] + pair[1, 2]))

    if show_atoms:

        # Draw circles
        for i in range(cell.size):

            if visible[i]:

                radius = e_size[i] * scale * circle_size

                g = svg.g()
                g.translate(*tuple(projected[i, :2]))
                if coordinates == 'right':
                    g.scale(1.0, -1.0)

                circle = svg.circle(
                    center=(0, 0),
                    r=radius,
                    fill=__svg_color__(colors_base[i]),
                    stroke=__svg_color__(__dark__(colors_base[i])),
                    stroke_width=circle_stroke * radius,
                )

                if not circle_opacity is None:
                    if isinstance(circle_opacity, (int, float)):
                        circle.fill(opacity=circle_opacity)
                        circle.stroke(opacity=circle_opacity)
                    else:
                        circle.fill(opacity=circle_opacity[i])
                        circle.stroke(opacity=circle_opacity[i])

                g.add(circle)

                if show_numbers:
                    g.add(svg.text(
                        str(i - 13 * N if invisible == "auto" else i),
                        insert=(0, radius * 0.35),
                        fill=__svg_color__(__dark__(colors_base[i])),
                        text_anchor="middle",
                        font_size=radius,
                        font_family=font_family,
                    ))

                obj.append(g)
                obj_z.append(projected[i, 2])

    d = cell.distances(cutoff=2*e_covsize.max())

    if show_bonds:

        # Draw lines
        for i in range(d.shape[0]):
            for j in range(i, d.shape[1]):
                if (visible[i] or visible[j]) and (0 < d[i, j] < bond_size * (e_covsize[i] + e_covsize[j])) and (
                        d[i, j] > (e_size[i] + e_size[j]) * circle_size):

                    unit = projected[j] - projected[i]
                    unit = unit / ((unit ** 2).sum()) ** 0.5

                    if show_atoms:
                        start = projected[i, :2] + unit[:2] * e_size[i] * circle_size * scale
                        end = projected[j, :2] - unit[:2] * e_size[j] * circle_size * scale

                    else:
                        start = projected[i, :2]
                        end = projected[j, :2]

                    start, end = __window__(start, end, (0, 0, osize[0], osize[1]))

                    if not start is None:
                        obj.append(svg.line(
                            start=start,
                            end=end,
                            stroke=__svg_color__((__dark__(colors_base[i]) + __dark__(colors_base[j])) / 2),
                            stroke_width=scale * (e_size[i] + e_size[j]) * circle_size / 5,
                        ))

                        obj_z.append((projected[j, 2] + projected[i, 2]) / 2)

    order = numpy.argsort(obj_z)
    for i in order[::-1]:
        subgroup.add(obj[i])

    if show_legend:

        unique = set(cell.values)

        __legend_margin__ = 10
        __box_size__ = (font_size + font_size_small) * 1.2
        __text_baseline__ = font_size * 0.2
        __i_x__ = font_size_small * 0.9
        __i_y__ = font_size_small * 1.2
        x = size[0] - (__legend_margin__ + __box_size__) * len(unique)
        y = __legend_margin__

        for i, e in enumerate(sorted(unique)):
            xx = x + (__legend_margin__ + __box_size__) * i
            yy = y

            color_base = element_color_convention[e.lower()]
            color_1 = __dark__(color_base, delta=0.8) if sum(color_base) > 0x180 else __light__(color_base, delta=0.8)

            group.add(svg.rect(
                insert=(xx, yy),
                size=(__box_size__, __box_size__),
                fill=__svg_color__(color_base),
                stroke_width=1,
                stroke=__svg_color__(color_1),
                rx=2,
                ry=2,
            ))

            group.add(svg.text(
                str(element_number[e]),
                insert=(xx + __i_x__, yy + __i_y__),
                fill=__svg_color__(color_1),
                text_anchor="middle",
                **font_props_small
            ))

            group.add(svg.text(
                e[:1].upper() + e[1:],
                insert=(xx + __box_size__ / 2, yy + __box_size__ - __text_baseline__),
                fill=__svg_color__(color_1),
                text_anchor="middle",
                **font_props
            ))

    # This variable holds the background rectangle in case it spans the entire left side
    # To share it with the title
    left_side_box = False
    if show_vectors:
        __w__ = 4
        __h__ = 3
        __l__ = 20
        __offset__ = font_size * 0.9
        __offset2__ = font_size * 0.3
        __tmargin__ = font_size * 0.6
        __box_offset__ = 3
        __collapse_threshold__ = 0.2

        arrow_marker = svg.marker(
            insert=(__w__, __h__),
            size=(2*__w__, 2*__h__),
            orient="auto",
            markerUnits="strokeWidth",
        )
        arrow_marker.add(svg.path(d="M0,0 L0,{:d} L{:d},{:d} z".format(2*__h__, 2*__w__, __h__), fill="black"))
        svg.defs.add(arrow_marker)

        a_group = svg.g()
        subgroup.add(a_group)

        pvecs = projection.transform_from_cartesian(cell.vectors)

        pvecs_n = pvecs / numpy.linalg.norm(pvecs, axis=-1)[:, numpy.newaxis]
        pvecs_rxy = numpy.linalg.norm(pvecs_n[:, :2], axis=-1)

        pvecs_selection = pvecs_rxy > __collapse_threshold__

        pvecs_n = pvecs_n[pvecs_selection, :]
        pvecs_label = tuple(i for i, j in zip("xyz", pvecs_selection) if j)
        pvecs_rxy = pvecs_rxy[pvecs_selection]
        pvecs_nxy = pvecs_n[:, :2] / pvecs_rxy[:, numpy.newaxis]
        # Label location candidates:
        # To the left of the axis
        pvecs_llc = pvecs_nxy.dot([[0, 1], [-1, 0]]) * __offset__  + pvecs_n[:, :2] * __l__ / 2
        # To the right of the axis
        pvecs_llc2 = pvecs_nxy.dot([[0, -1], [1, 0]]) * __offset__  + pvecs_n[:, :2] * __l__ / 2
        # On top of the vector
        pvecs_llc3 = pvecs_n[:, :2] * __l__ + pvecs_nxy * __offset__
        pvecs_llc = numpy.concatenate((pvecs_llc[:, numpy.newaxis, :], pvecs_llc2[:, numpy.newaxis, :], pvecs_llc3[:, numpy.newaxis, :]), axis=1)

        a = numpy.arctan2(pvecs_n[:, 1], pvecs_n[:, 0])
        correct_order = numpy.argsort(a)
        # Roll the order until it starts with zero
        correct_order = numpy.roll(correct_order, -numpy.argmin(correct_order))


        best = None
        best_d = None

        for llc in product(*pvecs_llc):
            llc = numpy.array(llc)
            a = numpy.arctan2(*llc[:, ::-1].T)
            trial_order = numpy.argsort(a)
            trial_order = numpy.roll(trial_order, -numpy.argmin(trial_order))
            if numpy.all(trial_order == correct_order):
                d = numpy.linalg.norm(llc[numpy.newaxis, :, :] - llc[:, numpy.newaxis, :], axis=-1)
                wf = abs(d - __l__ * 1.4).sum()
                if best is None or wf < best_d:
                    best_d = wf
                    best = llc
        if best is not None:
            pvecs_llc = best
        else:
            pvecs_llc = llc

        obj = []
        obj_z = []
        boxes = []

        for (x, y, z), (nx, ny), l, (tx, ty) in zip(pvecs_n, pvecs_nxy, pvecs_label, pvecs_llc):
            line = svg.line((0, 0), (__l__, 0), stroke="black", marker_end=arrow_marker.get_funciri())
            line.matrix(x, y, -ny, nx, 0, 0)

            text = svg.text(
                l,
                insert=(tx, ty),
                text_anchor="middle",
                **font_props
            )

            if coordinates == 'right':
                text.matrix(1, 0, 0, -1, 0, 2*ty - __offset2__)

            obj.append(line)
            obj_z.append(z)

            obj.append(text)
            obj_z.append(z)

            boxes.extend((
                (0, 0),
                (x * (__l__ + __w__), y * (__l__ + __w__)),
                (tx - __tmargin__, ty - __tmargin__),
                (tx + __tmargin__, ty + __tmargin__)
            ))

        boxes = numpy.array(boxes)
        bmin = boxes.min(axis=0) - __box_offset__
        if vectors_offset is not None:
            bmin -= vectors_offset
        bmax = boxes.max(axis=0) + __box_offset__
        bwh = bmax - bmin
        if bwh[0] > osize[0] / 2:
            bwh[0] = osize[0] + margin
        if bwh[1] > osize[1] / 2:
            bwh[1] = osize[1] + margin
            left_side_box = True

        rect = svg.rect(
            insert=bmin - margin,
            size=bwh + margin,
            fill="white",
            opacity=overlay_opacity,
        )
        if left_side_box:
            left_side_box = rect
        a_group.add(rect)
        a_group.translate(*(-bmin))

        order = numpy.argsort(obj_z)
        for i in order[::-1]:
            a_group.add(obj[i])

    if title is not None:
        __text_margin__ = font_size * 0.25
        __w__ = font_size * len(title)
        __h__ = font_size

        g = svg.g()
        if not left_side_box:
            g.add(svg.rect(
                insert=(-margin, -margin),
                size=(__w__ + margin, __h__ + margin),
                fill="white",
                opacity=overlay_opacity,
            ))
        else:
            left_side_box["width"] = max(left_side_box["width"], __w__ + margin)

        g.add(svg.text(
            title,
            insert=(__w__ * 0.5, __h__ - __text_margin__),
            fill="black",
            text_anchor="middle",
            **font_props
        ))
        g.translate(margin, margin)
        group.add(g)

    if save:
        svg.save()

    if return_coords:
        return svg, (subgroup, (projected[13*N : 14*N] if invisible=="auto" else projected)[:, :2])

    else:
        return svg


def __guess_energy_range__(cell, bands=10, window=0.05, center_fermi=True):
    """
    Attempts to guess the energy range of interest.

    Args:

        cell (Cell): cell with the band structure;
        bands (int): number of bands to focus;
        window (float): relative size of the gaps below and above
        selected energy range;
        center_fermi (bool): if True, centers the Fermi level within the window;

    Returns:
        A tuple with the energy range.
    """
    if cell.fermi is not None and cell.values.shape[1] > bands:

        minimas = cell.values.min(axis=0)
        maximas = cell.values.max(axis=0)

        top = numpy.argsort(numpy.maximum(
            numpy.abs(minimas - cell.fermi),
            numpy.abs(maximas - cell.fermi),
        ))

        global_min = minimas[top[:bands]].min()
        global_max = maximas[top[:bands]].max()

        if center_fermi:
            shift = .5 * (global_min + global_max) - cell.fermi
            global_min -= shift
            global_max -= shift

    else:

        global_min = cell.values.min()
        global_max = cell.values.max()

    return numpy.array((
        ((1 + window) * global_min - window * global_max),
        ((1 + window) * global_max - window * global_min),
    ))


def matplotlib_bands(
        cell,
        axes,
        show_fermi=True,
        fermi_origin=False,
        energy_range=None,
        energy_units="eV",
        energy_units_name=None,
        coordinate_units=None,
        coordinate_units_name=None,
        threshold=1e-2,
        weights=None,
        weights_color=None,
        weights_size=None,
        optimize_visible=False,
        edge_names=None,
        mark_points=None,
        project=None,
        return_projected=False,
        ls="-",
        **kwargs
):
    """
    Plots basic band structure using pyplot.

    Args:
        cell (BandsPath): cell with the band structure;
        axes (matplotlib.axes.Axes): axes to plot on;
        show_fermi (bool): shows the Fermi level if specified;
        fermi_origin (bool): shift the energy origin to the Fermi level;
        energy_range (array): 2 floats defining plot energy range. The
        units of energy are defined by the ``units`` keyword;
        energy_units (str, float): either a field from ``numericalunits``
        package or a float with energy units;
        energy_units_name (str): a string used for the units. Used only if the
        ``energy_units`` keyword is a float;
        coordinate_units (str, float): either a field from ``numericalunits``
        package or a float with coordinate units or None;
        coordinate_units_name (str): a string used for the coordinate
        units. Used only if the ``coordinate_units`` keyword is a float;
        threshold (float): threshold for determining edges of k point
        path;
        weights, weights_color (array): a 2D array with weights on the
        band structure which will be converted to color according to
        current colormap;
        weights_size (array): a 2D array with weights on the band
        structure which will be converted to line thickness;
        optimize_visible (bool): draw only visible lines;
        edge_names (list): the edges names to appear on the band structure;
        mark_points (list): marks specific points on the band structure,
        the first number in each list element is interpreted as k-point
        while the second number is band number;
        project (array): projects k-points along specified direction
        instead of unfolding the entire bands path. If ``coordinate_units``
        specified the direction is expressed in the unit cell vectors,
        otherwise cartesian basis is used;
        return_projected (bool): if True, additionally returns a 1D array
        with x coordinates of bands on the plot;
        ls (str): a shortcut for line styles: "-", "--", ".", "-.";
        The rest of kwargs are passed to
        ``matplotlib.collections.LineCollection``.

    Returns:

        A plotted LineCollection.
    """

    from matplotlib.collections import LineCollection
    from matplotlib.transforms import blended_transform_factory

    if not weights is None and not (cell.values.shape == weights.shape):
        raise TypeError(
            "The shape of 'weights' {} is different from the shape of band structure data {}".format(weights.shape,
                                                                                                     cell.values.shape))

    if not weights_color is None and not (cell.values.shape == weights_color.shape):
        raise TypeError("The shape of 'weights_color' {} is different from the shape of band structure data {}".format(
            weights_color.shape, cell.values.shape))

    if not weights_size is None and not (cell.values.shape == weights_size.shape):
        raise TypeError("The shape of 'weights_size' {} is different from the shape of band structure data {}".format(
            weights_size.shape, cell.values.shape))

    if not weights is None and weights_color is None:
        weights_color = weights

    if isinstance(energy_units, str):
        energy_units_name = energy_units
        energy_units = getattr(numericalunits, energy_units)

    if isinstance(coordinate_units, str):
        coordinate_units_name = coordinate_units
        coordinate_units = getattr(numericalunits, coordinate_units)

    # Move the origin to the Fermi level
    if fermi_origin and cell.fermi is not None:
        cell = cell.canonize_fermi()

    # Set energy range
    if energy_range is None:
        energy_range = __guess_energy_range__(cell) / energy_units

    if edge_names is None:
        edge_names = tuple()

    defaults = {}
    if ls == "-":
        defaults.update(dict(
            capstyle="round",
            joinstyle="round",
            linestyles="solid",
        ))
    elif ls == "--":
        defaults["linestyles"] = (0, (1, 2))
    elif ls in ".:":
        defaults["linestyles"] = "dotted"
    elif ls == "-.":
        defaults["linestyles"] = "dashdot"
    else:
        raise ValueError("Unknown line style: {}".format(ls))

    defaults.update(kwargs)
    kwargs = defaults

    if "color" in kwargs:
        kwargs["colors"] = kwargs["color"]
        del kwargs["color"]

    # Cycle color
    if not "colors" in kwargs:
        kwargs.update(next(axes._get_lines.prop_cycler))

    if cell.size > 1:

        # Fold K points to 0- > 1 line or project
        if not project is None:
            __ = {"kx": 0, "x": 0, "ky": 1, "y": 1, "kz": 2, "z": 2}
            if project in __.keys():
                x_label = project
                v = [0] * cell.vectors.shape[0]
                v[__[project]] = 1
                project = v

            else:
                x_label = ("(" + (",".join(("{:.2f}",) * len(project))) + ") direction").format(*project)

            project = numpy.array(project, dtype=numpy.float)
            project /= (project ** 2).sum() ** .5

            if coordinate_units is None:
                kpoints = numpy.dot(cell.coordinates, project)

            else:
                kpoints = numpy.dot(cell.cartesian(), project) / coordinate_units

        else:
            x_label = None

            kpoints = cell.distances((0,) + tuple(range(cell.size)))
            for i in range(1, kpoints.shape[0]):
                kpoints[i] += kpoints[i - 1]

            if coordinate_units is None:
                kpoints /= kpoints[-1]

            else:
                kpoints /= coordinate_units

        if not coordinate_units_name is None:
            if x_label is None:
                x_label = "(" + coordinate_units_name + ")"

            else:

                x_label += " (" + coordinate_units_name + ")"

        # Find location of edges on the K axis
        if cell.size > 2:
            makes_turn = numpy.abs(1. + cell.angles(range(cell.size))) > threshold
            makes_turn = numpy.concatenate([[True], makes_turn, [True]])
        else:
            makes_turn = numpy.array([True, True])
        edges = kpoints[makes_turn]

        # Plot edges
        if project is None:
            for e in edges[1:-1]:
                axes.axvline(x=e, color='black', linewidth=0.5)

        # Get continious parts
        continious = numpy.logical_not(makes_turn[1:] * makes_turn[:-1])

        # Take care of isolated points
        discontinious = numpy.logical_not(numpy.concatenate(([False], continious, [False])))
        isolated_points = discontinious[1:] * discontinious[:-1]
        continious = numpy.logical_or(continious, numpy.logical_or(isolated_points[1:], isolated_points[:-1]))

        # Get the segments to draw
        visible_segment = continious[:, numpy.newaxis] * numpy.ones((1, cell.values.shape[1]), dtype=numpy.bool)

        # Optimize visible segments
        if optimize_visible:
            visible_point = numpy.logical_and(cell.values / energy_units > energy_range[0],
                                              cell.values / energy_units < energy_range[1])
            visible_segment = numpy.logical_and(
                numpy.logical_or(visible_point[:-1, :], visible_point[1:, :]),
                visible_segment
            )

        # Prepare LineCollection
        segment_sets = []
        for i in range(cell.values.shape[1]):
            points = numpy.array([kpoints, cell.values[:, i] / energy_units]).T.reshape(-1, 1, 2)
            segments = numpy.concatenate([points[:-1][visible_segment[:, i]], points[1:][visible_segment[:, i]]],
                                         axis=1)

            segment_sets.append(segments)

        segments = numpy.concatenate(segment_sets, axis=0)

        lc = LineCollection(segments, **kwargs)

        # Weights
        for array, target in ((weights_color, lc.set_array), (weights_size, lc.set_linewidth)):
            if not array is None:
                array = numpy.swapaxes(0.5 * (array[1:, :] + array[:-1, :]), 0, 1)
                array = array[numpy.swapaxes(visible_segment, 0, 1)]
                target(array)

        # Mark points
        if not mark_points is None:
            mark_points = numpy.array(mark_points)
            axes.scatter(
                list(kpoints[i] for i, j in mark_points),
                list(cell.values[i, j] / energy_units for i, j in mark_points),
                marker="+",
                s=50,
            )

        if project is None:
            axes.set_xticks(edges)
            axes.set_xticklabels(list(
                edge_names[i] if i < len(edge_names) else " ".join(("{:.2f}",) * cell.coordinates.shape[1]).format(
                    *cell.coordinates[makes_turn, :][i])
                for i in range(makes_turn.sum())
            ))

        axes.set_xlim((kpoints.min(), kpoints.max()))
        if not x_label is None:
            axes.set_xlabel(x_label)

    else:

        kwargs.update(transform=blended_transform_factory(axes.transAxes, axes.transData))
        lc = LineCollection(list([[0, v], [1, v]] for v in cell.values.reshape(-1) / energy_units), **kwargs)

        axes.set_xticks([])

    # Plot bands
    axes.add_collection(lc)

    # Plot Fermi energy
    if show_fermi and cell.fermi is not None:
        axes.axhline(y=cell.fermi / energy_units, color='black', ls="--", lw=0.5)

    axes.set_ylim(energy_range)

    if not energy_units_name is None:
        axes.set_ylabel('Energy ({})'.format(energy_units_name))

    else:
        axes.set_ylabel('Energy')

    if return_projected:
        return lc, kpoints
    else:
        return lc


def matplotlib_bands_density(
        cell,
        axes,
        energies,
        show_fermi=True,
        fermi_origin=False,
        energy_range=None,
        units="eV",
        units_name=None,
        weights=None,
        on_top_of=None,
        use_fill=False,
        orientation="landscape",
        gaussian_spread=None,
        method="optimal",
        postproc=None,
        **kwargs
):
    """
    Plots density of bands (density of states).
    The cell values are considered to be band energies.

    Args:
        cell (BandsPath, BandsGrid): a unit cell with the band structure,
        possibly on the grid;
        axes (matplotlib.axes.Axes): axes to plot on;
        energies (int,array): energies to calculate density at. The
        integer value has the meaning of number of points to cover
        the range ``energy_range``. Otherwise the units of energy are
        defined by the ``units`` keyword;
        show_fermi (bool): shows the Fermi level if specified;
        fermi_origin (bool): shift the energy origin to the Fermi level;
        energy_range (array): 2 floats defining plot energy range. The
        units of energy are defined by the ``units`` keyword;
        units (str, float): either a field from ``numericalunits``
        package or a float with energy units;
        units_name (str): a string used for the units. Used only if the
        ``units`` keyword is a float;
        weights (array): a 2D array with weights on the band structure;
        on_top_of (array): a 2D array with weights on the band structure
        to plot on top of;
        use_fill (bool): fill the area below plot;
        orientation (str): either 'portrait' or 'landscape' - orientation
        of the plot;
        gaussian_spread (float): the gaussian spread for the density of
        states. This value is used only if the provided ``cell`` is not
        a Grid;
        method (str): method to calculate density: 'default', 'gaussian'
        or 'optimal';
        postproc (Callable): a post-processing function accepting density
        and energy values (in final units) and returning density values;
        The rest of kwargs are passed to pyplot plotting functions.

    Returns:

        A plotted Line2D or a PolyCollection, depending on ``use_fill``.
    """

    if not orientation == "portrait" and not orientation == "landscape":
        raise ValueError("Unknown orientation: {}".format(orientation))

    if isinstance(units, str):
        units_name = units
        units = getattr(numericalunits, units)

    # Move the origin to the Fermi level
    if fermi_origin and cell.fermi is not None:
        cell = cell.canonize_fermi()

    # Set energy range
    if energy_range is None:
        if isinstance(energies, numpy.ndarray):
            energy_range = energies[0], energies[-1]
        else:
            energy_range = __guess_energy_range__(cell) / units

    if isinstance(energies, int):
        energies = numpy.linspace(energy_range[0], energy_range[1], energies)
    else:
        energies = numpy.array(energies, dtype=numpy.float64)

    if weights is None:
        weights = 1

    if not isinstance(weights, numpy.ndarray):
        weights = weights * numpy.ones(cell.values.shape, dtype=numpy.float64)

    if on_top_of is None:
        on_top_of = numpy.zeros(cell.values.shape, dtype=numpy.float64)

    # Try converting to grid
    if method == "optimal":
        method = "gaussian"
        if cell.size == 1:
            pass
        elif isinstance(cell, Grid):
            method = "tetrahedron"
        elif isinstance(cell, Cell):
            grid = cell.as_grid()
            if grid.size == cell.size:
                cell = grid
                method = "tetrahedron"
                weights = numpy.reshape(weights, grid.values.shape)
                on_top_of = numpy.reshape(on_top_of, grid.values.shape)

    # Calculate DoS using tetrahedron method ...
    if isinstance(cell, Grid) and method == 'tetrahedron':

        data = cell.tetrahedron_density(energies * units, resolved=False, weights=weights)
        data_baseline = cell.tetrahedron_density(energies * units, resolved=False, weights=on_top_of)

    # ... or point-based method
    else:
        if method == "gaussian":

            if gaussian_spread is None:
                gaussian_spread = (energies.max() - energies.min()) / len(energies)
            _A = -0.5 / (gaussian_spread * units) ** 2
            _B = 1 / (2 * math.pi) ** 0.5 / (gaussian_spread * units)

            def method(x):
                return _B * numpy.exp(_A * x ** 2)

        elif not callable(method):
            raise ValueError("Method is not a callable: {}".format(repr(method)))

        _values = cell.values.reshape(-1)[numpy.newaxis, :]
        _weights = weights.reshape(-1)[numpy.newaxis, :]
        _on_top_of = on_top_of.reshape(-1)[numpy.newaxis, :]
        _energies = energies[:, numpy.newaxis] * units

        data = (_weights * method(_values - _energies)).sum(axis=-1) / cell.size
        data_baseline = (_on_top_of * method(_values - _energies)).sum(axis=-1) / cell.size

    data += data_baseline
    data *= units
    data_baseline *= units

    if postproc is not None:
        data = postproc(data, energies)

    if "color" not in kwargs:
        kwargs.update(next(axes._get_lines.prop_cycler))

    if orientation == "portrait":

        if use_fill:
            plot = axes.fill_betweenx(energies, data, data_baseline, **kwargs)
        else:
            plot = axes.plot(data, energies, **kwargs)

        if cell.fermi is not None and show_fermi:
            axes.axhline(y=cell.fermi / units, color='black', ls="--", lw=0.5)

        axes.set_ylim(energy_range)

        if not units_name is None:
            axes.set_xlabel('Density (electrons per unit cell per {})'.format(units_name))
            axes.set_ylabel('Energy ({})'.format(units_name))

        else:
            axes.set_xlabel('Density')
            axes.set_ylabel('Energy')

    elif orientation == "landscape":

        if use_fill:
            plot = axes.fill_between(energies, data, data_baseline, **kwargs)
        else:
            plot = axes.plot(energies, data, **kwargs)

        if cell.fermi is not None and show_fermi:
            axes.axvline(x=cell.fermi / units, color='black', ls="--", lw=0.5)

        axes.set_xlim(energy_range)

        if not units_name is None:
            axes.set_ylabel('Density (states per unit cell per {})'.format(units_name))
            axes.set_xlabel('Energy ({})'.format(units_name))

        else:
            axes.set_ylabel('Density')
            axes.set_xlabel('Energy')

    return plot


def __covering_range__(left, right, spacing, anchor=None):
    left, right = min(left, right), max(left, right)
    spacing = abs(spacing)
    if anchor is None:
        anchor = left
    # anchor = anchor % spacing
    left_scaled = numpy.floor((left - anchor) / spacing)
    right_scaled = numpy.ceil((right - anchor) / spacing)
    return numpy.arange(left_scaled, right_scaled + 1) * spacing + anchor


def matplotlib_scalar(
        grid,
        axes,
        origin,
        plane,
        units="angstrom",
        units_name=None,
        show_cell=False,
        normalize=True,
        ppu=None,
        isolines=None,
        window=None,
        margins=0.1,
        scale_bar=None,
        scale_bar_location=4,
        scale_bar_color="black",
        postproc=None,
        **kwargs
):
    """
    Plots scalar values on the grid using imshow.

    Args:
        grid (Grid): a 3D grid to be plotted;
        axes (matplotlib.axes.Axes): axes to plot on;
        origin (array): origin of the 2D slice to be plotted in the
        units of ``grid``;
        plane (str, int): the plotting plane: either 'x','y' or 'z' or
        the corresponding int.
        units (str, float): either a field from ``numericalunits``
        package or a float with axes units;
        units_name (str): a string used for the units. Used only if the
        ``units`` keyword is a float;
        show_cell (bool): if True then projected unit cell boundaries are
        shown on the final image;
        normalize (bool): normalize data before plotting such that the
        minimum is set at zero and the maximum is equal to one;
        ppu (float): points per ``unit`` for the raster image;
        isolines (array): plot isolines at the specified levels;
        window (array): 4 values representing a window to plot the data:
        minimum and maximum 'x' coordinate and minimum and maximum 'y'
        coordinate;
        margins (float): adds margins to the grid where the data is
        interpolated;
        scale_bar (float): adds a scale bar of the specified width to the image;
        scale_bar_location (int): location of the scale bar;
        scale_bar_color (str): the color of the scale bar;
        postproc (Callable): a callable to post-process the interpolated data;
        **kwargs: passed to ``pyplot.imshow`` or ``pyplot.contour``;

    Returns:
        A ``matplotlib.image.AxesImage`` plotted.
    """
    if grid.vectors.shape[0] != 3:
        raise ValueError("A {:d}D grid found, required 3D".format(grid.vectors.shape[0]))

    if isinstance(grid, Grid):
        if grid.values.ndim != 3 and isolines is None:
            raise ValueError("For color plot [isolines=None], the number of data dimensions "
                             "of the grid should be 3, found: {:d}".format(grid.values.ndim))

    elif isinstance(grid, Cell):
        if grid.values.ndim != 1 and isolines is None:
            raise ValueError("For color plot [isolines=None], the number of data dimensions "
                             "of the cell should be 1, found: {:d}".format(grid.values.ndim))

    else:
        raise ValueError("Unknown grid input: {}".format(grid))

    if isinstance(units, str):
        units_name = units
        units = getattr(numericalunits, units)

    origin = grid.transform_to_cartesian(origin)[numpy.newaxis, :]

    plane = __xyz2i__[plane]
    otherVectors = list(range(3))
    del otherVectors[plane]

    # Build a rotated cartesian basis
    v1 = grid.vectors[otherVectors][0]
    v3 = grid.vectors[plane]
    v2 = numpy.cross(v3, v1)
    basis = Basis((v1, v2, v3))
    basis = Basis(basis.vectors / basis.vectors_len[:, None])

    # Calculate in-plane coordinates of the grid edges
    edges_inplane = basis.transform_from_cartesian(grid.vertices - origin)
    if window is None:
        mn = edges_inplane.min(axis=0)
        mx = edges_inplane.max(axis=0)
    else:
        mn = numpy.array((window[0], window[2])) * units
        mx = numpy.array((window[1], window[3])) * units

    # Margins
    mn_a, mx_a = mn, mx
    mn, mx = mn * (1 + margins) + mx * (-margins), mn * (-margins) + mx * (1 + margins)

    if ppu is None:
        ppu = (grid.size / grid.volume) ** (1. / 3)

    else:
        ppu /= units

    # In-plane grid size: px, py
    px = int(round((mx[0] - mn[0]) * ppu))
    py = int(round((mx[1] - mn[1]) * ppu))
    if px * py == 0:
        raise ValueError(
            "The data is too sparse: the suggested ppu is {:e} points per {:s} while grid dimensions are {:e} and {:e} {:s}. Please set the ppu parameter manually".format(
                ppu * units,
                units_name,
                (mx[0] - mn[0]) / units,
                (mx[1] - mn[1]) / units,
                units_name,
            ))

    # Build an inplane grid
    upp = 1. / ppu
    x = __covering_range__(mn[0], mx[0], upp, anchor=0)
    y = __covering_range__(mn[1], mx[1], upp, anchor=0)
    mg = numpy.meshgrid(x, y, (0,), indexing='ij')
    dims = mg[0].shape[:2]
    points_inplane = numpy.concatenate(tuple(i[..., numpy.newaxis] for i in mg), axis=len(mg)).reshape(-1, 3)

    # Convert to lattice coordinates of the initial grid
    points_cartesian = basis.transform_to_cartesian(points_inplane) + origin
    points_lattice = grid.transform_from_cartesian(points_cartesian)

    # Interpolate
    if isinstance(grid, Grid):
        interpolated = grid.interpolate_to_cell(points_lattice)

    else:
        interpolated = grid.interpolate(points_lattice)

    interpolated_values = interpolated.values.copy()
    if interpolated_values.ndim == 1:
        interpolated_values = interpolated_values.reshape(*dims)
    else:
        interpolated_values = interpolated_values.reshape(*(dims + (-1,)))

    if postproc is not None:
        interpolated_values = postproc(interpolated_values)

    image = None

    if isolines is None:

        if normalize:
            interpolated_values -= interpolated_values.min()
            interpolated_values /= interpolated_values.max()

        image = axes.imshow(numpy.swapaxes(interpolated_values, 0, 1), extent=[
            (min(x) - upp/2) / units,
            (max(x) + upp/2) / units,
            (min(y) - upp/2) / units,
            (max(y) + upp/2) / units,
        ], origin="lower", **kwargs)

    else:

        interpolated_values = numpy.swapaxes(interpolated_values, 0, 1)
        if interpolated_values.ndim == 2:
            interpolated_values = interpolated_values[..., numpy.newaxis]
        lmax = max(isolines)
        lmin = min(isolines)
        for i in range(interpolated_values.shape[-1]):
            if interpolated_values[..., i].min() < lmax and interpolated_values[..., i].max() > lmin:
                image = axes.contour(x / units, y / units, interpolated_values[..., i], isolines, **kwargs)
        axes.set_aspect('equal')

    if show_cell:

        edges = basis.transform_from_cartesian(grid.edges - origin) / units
        for e in edges:
            axes.plot([e[0, 0], e[1, 0]], [e[0, 1], e[1, 1]], color="black")

    axes.set_xlim([mn_a[0] / units, mx_a[0] / units])
    axes.set_ylim([mn_a[1] / units, mx_a[1] / units])

    if not units_name is None:
        axes.set_xlabel("x ({})".format(units_name))
        axes.set_ylabel("y ({})".format(units_name))
    else:
        axes.set_xlabel("x")
        axes.set_ylabel("y")

    if scale_bar is not None:

        from matplotlib.patches import Rectangle

        t1 = axes.transData
        t2 = axes.transAxes
        t = t2 - t1

        if scale_bar_location == 1:
            x, y = .9, .9
            w, h = -1, -.05

        elif scale_bar_location == 2:
            x, y = .1, .9
            w, h = 1, -.05

        elif scale_bar_location == 3:
            x, y = .1, .1
            w, h = 1, .05

        elif scale_bar_location == 4:
            x, y = .9, .1
            w, h = -1, .05

        else:
            raise ValueError("Unknown location for the scale bar: {:r}".format(scale_bar_location))

        ((x, y), (_, h)) = t.transform(((x, y), (w, y + h)))
        h -= y
        w = (scale_bar / units) * w
        axes.add_patch(Rectangle((x, y), w, h, color=scale_bar_color, zorder=100))

    return image


def matplotlib2svgwrite(fig, svg, insert, size=None, method="firm", image_format=None, **kwargs):
    """
    Saves a matplotlib image to an existing svgwrite object.

    Args:

        fig (matplotlib.figure.Figure): a figure to save;

        svg (svgwrite.Drawing): an svg drawing to save to;

        insert (tuple): a tuple of ints defining destination to insert
        a drawing;

        size (tuple): size of the inserted image;

        method (str): the embedding method: either 'loose' (the plot is
        rasterized) or 'firm' (the plot's svg is embedded via <svg> tag);

    Kwargs:

        The kwargs are passed to ``fig.savefig`` used to print the plot.
    """
    if image_format is None:
        image_format = dict(loose="png", firm="svg")[method]
    if method == "firm" and image_format != "svg":
        raise ValueError("Only SVG images can be embedded with the 'firm' method")
    if method == "firm" and svg._parameter.profile != "full":
        raise ValueError("'firm' method requires a full svg profile")

    image_bin = StringIO()
    fig.savefig(image_bin, format=image_format, **kwargs)
    image_bin.seek(0)

    if method == "loose":
        image_str = "data:image/{};base64,".format(image_format) + base64.b64encode(image_bin.read())
        svg.add(svg.image(
            image_str,
            insert=insert,
            size=size,
        ))

    elif method == "firm":
        root = ElementTree.fromstring(image_bin.read())
        root.attrib["x"], root.attrib["y"] = map(str, insert)
        if size is not None:
            root.attrib["width"], root.attrib["height"] = map(str, size)
        esvg = svg.g()
        esvg.get_xml = lambda: root
        svg.add(esvg)

    else:
        raise ValueError("Illegal 'method' value")


def notebook_unit_cell(cell, **kwargs):
    """Display unit cell in iPython notebook."""
    from IPython.display import SVG, display
    f = NamedTemporaryFile()
    svgwrite_unit_cell(cell, f.name, **kwargs)
    display(SVG(filename=f.name))
