"""
This submodule contains data visualization routines.
"""
from .types import Basis, UnitCell, Grid, __angle__, __xyz2i__

import math
import colorsys
import base64
from StringIO import StringIO

import numpy
import numericalunits

__elements_table__ = (
    ('H' , (255,255,255), 0.53, 0.37),
    ('He', (217,255,255), 0.31, 0.32),
    ('Li', (204,128,255), 1.67, 1.34),
    ('Be', (194,255,  0), 1.12, 0.90),
    ('B' , (255,181,181), 0.87, 0.82),
    ('C' , (144,144,144), 0.67, 0.77),
    ('N' , ( 48, 80,248), 0.56, 0.75),
    ('O' , (255, 13, 13), 0.48, 0.73),
    ('F' , (144,224, 80), 0.42, 0.71),
    ('Ne', (179,227,245), 0.38, 0.69),
    ('Na', (171, 92,242), 1.90, 1.54),
    ('Mg', (138,255,  0), 1.45, 1.30),
    ('Al', (191,166,166), 1.18, 1.18),
    ('Si', (240,200,160), 1.11, 1.11),
    ('P' , (255,128,  0), 0.98, 1.06),
    ('S' , (255,255, 48), 0.88, 1.02),
    ('Cl', ( 31,240, 31), 0.79, 0.99),
    ('Ar', (128,209,227), 0.71, 0.97),
    ('K' , (143, 64,212), 2.43, 1.96),
    ('Ca', ( 61,255,  0), 1.94, 1.74),
    ('Sc', (230,230,230), 1.84, 1.44),
    ('Ti', (191,194,199), 1.76, 1.36),
    ('V' , (166,166,171), 1.71, 1.25),
    ('Cr', (138,153,199), 1.66, 1.27),
    ('Mn', (156,122,199), 1.61, 1.39),
    ('Fe', (224,102, 51), 1.56, 1.25),
    ('Co', (240,144,160), 1.52, 1.26),
    ('Ni', ( 80,208, 80), 1.49, 1.21),
    ('Cu', (200,128, 51), 1.45, 1.38),
    ('Zn', (125,128,176), 1.42, 1.31),
    ('Ga', (194,143,143), 1.36, 1.26),
    ('Ge', (102,143,143), 1.25, 1.22),
    ('As', (189,128,227), 1.14, 1.19),
    ('Se', (255,161,  0), 1.03, 1.16),
    ('Br', (166, 41, 41), 0.94, 1.14),
    ('Kr', ( 92,184,209), 0.88, 1.10),
    ('Rb', (112, 46,176), 2.65, 2.11),
    ('Sr', (  0,255,  0), 2.19, 1.92),
    ('Y' , (148,255,255), 2.12, 1.62),
    ('Zr', (148,224,224), 2.06, 1.48),
    ('Nb', (115,194,201), 1.98, 1.37),
    ('Mo', ( 84,181,181), 1.90, 1.45),
    ('Tc', ( 59,158,158), 1.83, 1.56),
    ('Ru', ( 36,143,143), 1.78, 1.26),
    ('Rh', ( 10,125,140), 1.73, 1.35),
    ('Pd', (  0,105,133), 1.69, 1.31),
    ('Ag', (192,192,192), 1.65, 1.53),
    ('Cd', (255,217,143), 1.61, 1.48),
    ('In', (166,117,115), 1.56, 1.44),
    ('Sn', (102,128,128), 1.45, 1.41),
    ('Sb', (158, 99,181), 1.33, 1.38),
    ('Te', (212,122,  0), 1.23, 1.35),
    ('I' , (148,  0,148), 1.15, 1.33),
    ('Xe', ( 66,158,176), 1.08, 1.30),
    ('Cs', ( 87, 23,143), 2.98, 2.25),
    ('Ba', (  0,201,  0), 2.53, 1.98),
    ('La', (112,212,255), 1.95, 1.69),
    ('Ce', (255,255,199), 1.85, 1.69),
    ('Pr', (217,255,199), 2.47, 1.69),
    ('Nd', (199,255,199), 2.06, 1.69),
    ('Pm', (163,255,199), 2.05, 1.69),
    ('Sm', (143,255,199), 2.38, 1.69),
    ('Eu', ( 97,255,199), 2.31, 1.69),
    ('Gd', ( 69,255,199), 2.33, 1.69),
    ('Tb', ( 48,255,199), 2.25, 1.69),
    ('Dy', ( 31,255,199), 2.28, 1.69),
    ('Ho', (  0,255,156), 2.26, 1.69),
    ('Er', (  0,230,117), 2.26, 1.69),
    ('Tm', (  0,212, 82), 2.22, 1.69),
    ('Yb', (  0,191, 56), 2.22, 1.69),
    ('Lu', (  0,171, 36), 2.17, 1.60),
    ('Hf', ( 77,194,255), 2.08, 1.50),
    ('Ta', ( 77,166,255), 2.00, 1.38),
    ('W' , ( 33,148,214), 1.93, 1.46),
    ('Re', ( 38,125,171), 1.88, 1.59),
    ('Os', ( 38,102,150), 1.85, 1.28),
    ('Ir', ( 23, 84,135), 1.80, 1.37),
    ('Pt', (208,208,224), 1.77, 1.28),
    ('Au', (255,209, 35), 1.74, 1.44),
    ('Hg', (184,184,208), 1.71, 1.49),
    ('Tl', (166, 84, 77), 1.56, 1.48),
    ('Pb', ( 87, 89, 97), 1.54, 1.47),
    ('Bi', (158, 79,181), 1.43, 1.46),
    ('Po', (171, 92,  0), 1.35, 1.46),
    ('At', (117, 79, 69), 1.27, 1.46),
    ('Rn', ( 66,130,150), 1.20, 1.45),
    ('Fr', ( 66,  0,102), 1.20, 1.45),
    ('Ra', (  0,125,  0), 1.20, 1.45),
    ('Ac', (112,171,250), 1.95, 1.45),
    ('Th', (  0,186,255), 1.80, 1.45),
    ('Pa', (  0,161,255), 1.80, 1.45),
    ('U' , (  0,143,255), 1.75, 1.45),
    ('Np', (  0,128,255), 1.75, 1.45),
    ('Pu', (  0,107,255), 1.75, 1.45),
    ('Am', ( 84, 92,242), 1.75, 1.45),
    ('Cm', (120, 92,227), 1.75, 1.45),
    ('Bk', (138, 79,227), 1.75, 1.45),
    ('Cf', (161, 54,212), 1.75, 1.45),
    ('Es', (179, 31,212), 1.75, 1.45),
    ('Fm', (179, 31,186), 1.75, 1.45),
    ('Md', (179, 13,166), 1.75, 1.45),
    ('No', (189, 13,135), 1.75, 1.45),
    ('Lr', (199,  0,102), 1.75, 1.45),
    ('Rf', (204,  0, 89), 1.75, 1.45),
    ('Db', (209,  0, 79), 1.75, 1.45),
    ('Sg', (217,  0, 69), 1.75, 1.45),
    ('Bh', (224,  0, 56), 1.75, 1.45),
    ('Hs', (230,  0, 46), 1.75, 1.45),
    ('Mt', (235,  0, 38), 1.75, 1.45),
)
__unknown_element__ = ('??', (0xA0,0xA0,0xA0), 1.75, 1.45)

__elements_name_lookup_table__ = dict((i[0].lower(), (n,)+i) for n, i in enumerate(__elements_table__))

def __fadeout_z__(color, z, mx, mn, strength, bg):
    
    alpha = min(max((mx - z)/(mx - mn)*strength,0),1)
    return (numpy.array(color, dtype = numpy.float64)*(1-alpha) + numpy.array(bg, dtype = numpy.float64)*alpha).astype(numpy.int64)
    
def __dark__(color, delta = 0.4):
    return (numpy.array(color, dtype = numpy.float64)*(1-delta)).astype(numpy.int64)
    
def __light__(color, delta = 0.4):
    return (255 - (255 - numpy.array(color, dtype = numpy.float64))*(1-delta)).astype(numpy.int64)
    
def __svg_color__(color):
    return "rgb({:d},{:d},{:d})".format(*color)
    
def svgwrite_unit_cell(
    cell,
    svg,
    camera = 'z',
    camera_top = None,
    insert = (0,0),
    size = (600,600),
    circle_size = 0.4,
    show_cell = False,
    show_atoms = True,
    show_bonds = True,
    show_legend = True,
    fadeout_strength = 0.8,
    bg = (0xF0,0xF0,0xFF),
):
    """
    Creates an svg drawing of a unit cell.
    
    Args:
    
        cell (UnitCell): the cell to be visualized;
        
        svg (str, svgwrite.Drawing): either file name to save the drawing
        to or an ``svgwrite.Drawing`` object to draw with.
        
    Kwargs:
    
        camera (str, array): the direction of a camera: either 'x','y' or
        'z' or an arbitrary 3D vector;
        
        camera_top (array): a vector pointing up;
        
        insert (array): a top-left corner of the drawing;
        
        size (array): size of the bounding box;
        
        circle_size (float): size of the circles representing atoms,
        arbitrary units;
        
        show_cell (bool): if True draws the unit cell edges projected;
        
        show_atoms (bool): if True draws atoms;
        
        show_bonds (bool): if True draws bonds;
        
        show_legend (bool): if True draws legend;
    
        fadeout_strength (float): amount of fadeout applied to more distant atoms;
        
        bg (array): an integer array defining background color;
        
    Returns:
    
        An ```svgwrite.Drawing`` object. The object is saved if it was
        created inside this method.
    """
    
    insert = numpy.array(insert, dtype = numpy.float64)
    size = numpy.array(size, dtype = numpy.float64)
    
    if isinstance(svg, str):
        import svgwrite
        save = True
        svg = svgwrite.Drawing(svg, size = size.tolist(), profile='tiny')
    else:
        save = False

    # Camera vector
    try:
        camera = {
            "x": (1,0,0),
            "y": (0,1,0),
            "z": (0,0,1),
        }[camera]
    except KeyError:
        pass     
    camera = numpy.array(camera, dtype = numpy.float64)
    
    # Camera top vector
    if camera_top is None:
        camera_top = numpy.array((0,0,1))
        if numpy.linalg.norm(numpy.cross(camera_top,camera)) == 0:
            camera_top = numpy.array((0,1,0))
    else:
        camera_top = numpy.array(camera_top, dtype = numpy.float64)
        if numpy.linalg.norm(numpy.cross(camera_top,camera)) == 0:
            raise ValueError("The 'camera' and 'camera_top' vectors cannot be collinear")
    
    # Calculate projection matrix
    projection = numpy.zeros((3,3), dtype = numpy.float64)
    projection[:,2] = camera
    projection[:,0] = numpy.cross(camera_top, projection[:,2])
    projection[:,1] = numpy.cross(projection[:,2], projection[:,0])
    
    projection /= ((projection**2).sum(axis = 1)**.5)[:,numpy.newaxis]
    
    # Project atomic coordinates onto the plane    
    projected = numpy.tensordot(cell.cartesian(),projection,axes = ((1,),(0,)))
    
    # Collect elements
    elements = tuple(__elements_name_lookup_table__[i.lower()] if i.lower() in __elements_name_lookup_table__ else (-1,) + __unknown_element__ for i in cell.values)
    e_color = tuple(i[2] for i in elements)
    e_size = numpy.array(tuple(i[3] for i in elements))*numericalunits.angstrom
    e_covsize = numpy.array(tuple(i[4] for i in elements))*numericalunits.angstrom
    
    # Determine boundaries
    b_min = numpy.min(projected - e_size[...,numpy.newaxis]*circle_size, axis = 0)
    b_max = numpy.max(projected + e_size[...,numpy.newaxis]*circle_size, axis = 0)
    
    if show_cell:
        
        # Project unit cell edges and 
        projected_edges = numpy.tensordot(cell.edges(),projection,axes = ((2,),(0,)))
        
        # Modify boundaries
        b_min = numpy.minimum(b_min, projected_edges.reshape(-1, projected_edges.shape[-1]).min(axis = 0))
        b_max = numpy.maximum(b_max, projected_edges.reshape(-1, projected_edges.shape[-1]).max(axis = 0))
    
    center = 0.5*(b_min + b_max)[:2]
    scale = (size/(b_max[:2]-b_min[:2])).min()
    shift = insert + 0.5*size - center*scale
    
    # Calculate base colors
    colors_base = tuple(__fadeout_z__(e_color[i], projected[i,2], b_max[2], b_min[2], fadeout_strength, bg) for i in range(cell.size()))
    
    # Arrays for storing objects with z-index
    obj = []
    obj_z = []
    
    if show_cell:
        
        # Draw unit cell edges
        for pair in projected_edges:
            obj.append(svg.line(
                start = pair[0][:2]*scale+shift,
                end = pair[1][:2]*scale+shift,
                stroke = "black",
                opacity = 0.1,
                stroke_width = 0.01*max(*size),
            ))
            obj_z.append(0.5*(pair[0,2] + pair[1,2]))
    
    if show_atoms:
        
        # Draw circles
        for i in range(cell.size()):
            
            radius = e_size[i]*scale*circle_size
            
            obj.append(svg.circle(
                center = projected[i,:2]*scale+shift,
                r = radius,
                fill = __svg_color__(colors_base[i]),
                stroke = __svg_color__(__dark__(colors_base[i])),
                stroke_width = 0.1*radius,
            ))
            
            obj_z.append(projected[i,2])
        
    d = cell.distances()
    
    if show_bonds:
        
        # Draw lines
        for i in range(d.shape[0]):
            for j in range(i,d.shape[1]):
                if (d[i,j]<(e_covsize[i]+e_covsize[j])) and (d[i,j]>(e_size[i]+e_size[j])*circle_size):
                    
                    unit = projected[j] - projected[i]
                    unit = unit / ((unit**2).sum())**0.5
                    
                    if show_atoms:
                        start = (projected[i,:2]+unit[:2]*e_size[i]*circle_size)*scale + shift
                        end = (projected[j,:2]-unit[:2]*e_size[j]*circle_size)*scale + shift
                    
                    else:
                        start = projected[i,:2]*scale + shift
                        end = projected[j,:2]*scale + shift
                    
                    obj.append(svg.line(
                        start = start,
                        end = end,
                        stroke = __svg_color__(__dark__((colors_base[i] + colors_base[j])/2)),
                        stroke_width = scale*(e_size[i]+e_size[j])*circle_size/5,
                    ))
                    
                    obj_z.append((projected[j,2] + projected[i,2])/2)
                
    svg.add(svg.rect(
        insert = insert,
        size = size,
        fill = __svg_color__(bg),
    ))
    
    order = numpy.argsort(obj_z)[::-1]
    for i in range(order.shape[0]):
        svg.add(obj[order[i]])
        
    if show_legend:
        
        unique = []
        for i in elements:
            if not i in unique:
                unique.append(i)
                
        __legend_margin__ = 10
        __box_size__ = 30
        __text_baseline__ = 5
        __text_size__ = 18
        __i_size__ = 10
        __i_x__ = 7
        __i_y__ = 10
        x = insert[0] + size[0] - (__legend_margin__ + __box_size__)*len(unique)
        y = insert[1] + __legend_margin__
        
        for i, e in enumerate(unique):
            
            xx = x + (__legend_margin__ + __box_size__)*i
            yy = y
            
            color_1 = __dark__(e[2], delta = 0.8) if sum(e[2])>0x180 else __light__(e[2], delta = 0.8)
                
            svg.add(svg.rect(
                insert = (xx,yy),
                size = (__box_size__, __box_size__),
                fill = __svg_color__(e[2]),
                stroke_width = 2,
                stroke = __svg_color__(color_1),
                rx = 2,
                ry = 2,
            ))
            
            svg.add(svg.text(str(e[0]+1),
                insert = (xx + __i_x__,yy + __i_y__),
                fill = __svg_color__(color_1),
                text_anchor = "middle",
                font_size = __i_size__,
            ))

            svg.add(svg.text(e[1],
                insert = (xx + __box_size__/2,yy + __box_size__ - __text_baseline__),
                fill = __svg_color__(color_1),
                text_anchor = "middle",
                font_size = __text_size__,
            ))
        
    if save:
        svg.save()
        
    return svg

def __guess_energy_range__(cell, bands = 10, window = 0.05):
    """
    Attempts to guess the energy range of interest.
    
    Args:
    
        cell (UnitCell): cell with the band structure;
    
    Kwargs:
    
        bands (int): number of bands to focus;
        
        window (float): relative size of the gaps below and above
        selected energy range;
        
    Returns:
    
        A tuple with the energy range.
    """
    if "Fermi" in cell.meta and cell.values.shape[1] > bands:
        
        minimas = cell.values.min(axis = 0)
        maximas = cell.values.max(axis = 0)
        
        top = numpy.argsort(numpy.maximum(
            numpy.abs(minimas - cell.meta["Fermi"]),
            numpy.abs(maximas - cell.meta["Fermi"]),
        ))
        
        global_min = minimas[top[:bands]].min()
        global_max = maximas[top[:bands]].max()
        
        return numpy.array((
            ((1+window)*global_min - window*global_max),
            ((1+window)*global_max - window*global_min),
        ))
        
    else:
        
        return numpy.array((cell.values.min(), cell.values.max()))
    
def matplotlib_bands(
    cell,
    axes,
    show_fermi = True,
    energy_range = None,
    units = "eV",
    units_name = None,
    threshold = 1e-2,
    weights = None,
    weights_color = None,
    weights_size = None,
    edge_names = [],
    **kwargs
):
    """
    Plots basic band structure using pyplot.
    
    Args:
    
        cell (UnitCell): cell with the band structure;
        
        axes (matplotlib.axes.Axes): axes to plot on;
        
    Kwargs:
        
        show_fermi (bool): shows the Fermi level if specified;
        
        energy_range (array): 2 floats defining plot energy range. The
        units of energy are defined by the ``units`` keyword;
        
        units (str, float): either a field from ``numericalunits``
        package or a float with energy units;
        
        units_name (str): a string used for the units. Used only if the
        ``units`` keyword is a float;
        
        threshold (float): threshold for determining edges of k point
        path;
        
        weights, weights_color (array): a 2D array with weights on the
        band structure which will be converted to color according to
        current colormap;
        
        weights_size (array): a 2D array with weights on the band
        structure which will be converted to line thickness;
        
        edge_names (list): the edges names to appear on the band structure;
        
        The rest of kwargs are passed to
        ``matplotlib.collections.LineCollection``.
        
    Returns:
    
        A plotted LineCollection.
    """
    
    from matplotlib.collections import LineCollection

    if not weights is None and not (cell.values.shape == weights.shape):
        raise TypeError("The shape of 'weights' {} is different from the shape of band structure data {}".format(weights.shape, cell.values.shape))
    
    if not weights_color is None and not (cell.values.shape == weights_color.shape):
        raise TypeError("The shape of 'weights_color' {} is different from the shape of band structure data {}".format(weights_color.shape, cell.values.shape))
        
    if not weights_size is None and not (cell.values.shape == weights_size.shape):
        raise TypeError("The shape of 'weights_size' {} is different from the shape of band structure data {}".format(weights_size.shape, cell.values.shape))
        
    if not weights is None and weights_color is None:
        weights_color = weights
    
    if isinstance(units, str):
        units_name = units
        units = getattr(numericalunits, units)
        
    # Fold K points to 0- > 1 line
    kpoints = cell.distances((0,)+tuple(range(cell.size())))
    for i in range(1,kpoints.shape[0]):
        kpoints[i] += kpoints[i-1]
    kpoints /= kpoints[-1]
    
    # Find location of edges on the K axis
    makes_turn = numpy.abs(1.+cell.angles(range(cell.size())))>threshold
    makes_turn = numpy.concatenate([[True], makes_turn, [True]])
    edges = kpoints[makes_turn]

    # Plot edges
    for i in range(edges.shape[0]):
        axes.axvline(x=edges[i],color='black',linewidth = 2)
        
    # Get continious parts
    continious = numpy.logical_not(makes_turn[1:]*makes_turn[:-1])
    
    # Prepare LineCollection
    segment_sets = []
    for i in range(cell.values.shape[1]):
        points = numpy.array([kpoints, cell.values[:,i]/units]).T.reshape(-1, 1, 2)
        segments = numpy.arange(cell.values.shape[0]-1)[continious]
        segments = numpy.concatenate([points[:-1][continious], points[1:][continious]], axis=1)
        segment_sets.append(segments)
        
    segments = numpy.concatenate(segment_sets,axis = 0)
    
    if not "color" in kwargs:
        kwargs["color"] = next(axes._get_lines.color_cycle)
        
    lc = LineCollection(segments, **kwargs)
    
    if not weights_color is None:
        a = 0.5*(weights_color[:-1][continious]+weights_color[1:][continious])
        lc.set_array(a.T.reshape(-1))
        
    if not weights_size is None:
        a = 0.5*(weights_size[:-1][continious]+weights_size[1:][continious])
        lc.set_linewidth(a.T.reshape(-1))
        
    # Plot bands
    axes.add_collection(lc)
    
    # Plot Fermi energy
    if show_fermi and "Fermi" in cell.meta:
        axes.axhline(y = cell.meta["Fermi"]/units,color='r')
        
    # Set energy range
    if energy_range is None:
        energy_range = __guess_energy_range__(cell)/units
                
    axes.set_ylim(energy_range)
    axes.set_xlim((0,1))
    
    axes.set_xticks(edges)
    axes.set_xticklabels(list(
        edge_names[i] if i<len(edge_names) else " ".join(("{:.2f}",)*cell.coordinates.shape[1]).format(*cell.coordinates[makes_turn,:][i])
        for i in range(makes_turn.sum())
    ))
    
    if not units_name is None:
        axes.set_ylabel('Energy, {}'.format(units_name))
        
    else:
        axes.set_ylabel('Energy')
    
    return lc

def matplotlib_bands_density(
    cell,
    axes,
    energies,
    show_fermi = True,
    energy_range = None,
    units = "eV",
    units_name = None,
    weights = None,
    on_top_of = None,
    use_fill = False,
    orientation = "landscape",
    gaussian_spread = None,
    force_gaussian = False,
    **kwargs
):
    """
    Plots density of bands (density of states).
    
    The cell values are considered to be band energies.
    
    Args:
    
        cell (Grid,UnitCell): a unit cell with the band structure,
        possibly on the grid;
        
        axes (matplotlib.axes.Axes): axes to plot on;
        
        energies (int,array): energies to calculate density at. The 
        integer value has the meaning of number of points to cover
        the range ``energy_range``. Otherwise the units of energy are
        defined by the ``units`` keyword;
        
    Kwargs:
    
        show_fermi (bool): shows the Fermi level if specified;
        
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
        
        force_gaussian (bool): use gaussians for the grids instead of a
        faster tetrahedron method;
        
        The rest of kwargs are passed to pyplot plotting functions.
        
    Returns:
    
        A plotted Line2D or a PolyCollection, depending on ``use_fill``.
    """
    
    if not orientation == "portrait" and not orientation == "landscape":
        raise ValueError("Unknown orientation: {}".format(orientation))
        
    if isinstance(units, str):
        units_name = units
        units = getattr(numericalunits, units)
    
    # Set energy range
    if energy_range is None:
        energy_range = __guess_energy_range__(cell)/units
        
    if isinstance(energies, int):
        energies = numpy.linspace(energy_range[0], energy_range[1], energies)
    else:
        energies = numpy.array(energies, dtype = numpy.float64)
    
    if weights is None:
        weights = numpy.ones(cell.values.shape, dtype = numpy.float64)
        
    if on_top_of is None:
        on_top_of = numpy.zeros(cell.values.shape, dtype = numpy.float64)
    
    # Calculate DoS using tetrahedron method ...
    if isinstance(cell, Grid) and not force_gaussian:
        
        grid = cell.tetrahedron_density(energies*units, resolved = True)
        
        data = (grid.values * weights[...,numpy.newaxis])
        for i in range(4):
            data = data.sum(axis = 0)
            
        data_baseline = (grid.values * on_top_of[...,numpy.newaxis])
        for i in range(4):
            data_baseline = data_baseline.sum(axis = 0)
    
    # ... or Gaussian
    else:
        if gaussian_spread is None:
            gaussian_spread = (energies.max() - energies.min())/len(energies)
        
        _values = cell.values.reshape(-1)[numpy.newaxis,:]
        _weights = weights.reshape(-1)[numpy.newaxis,:]
        _on_top_of = on_top_of.reshape(-1)[numpy.newaxis,:]
        _energies = energies[:,numpy.newaxis]*units
        _A = -0.5/(gaussian_spread*units)**2
        _B = 1/(2*math.pi)**0.5/(gaussian_spread*units)
        
        data = (_B*_weights*numpy.exp(_A*(_values-_energies)**2)).sum(axis = -1)/cell.size()
        data_baseline = (_B*_on_top_of*numpy.exp(_A*(_values-_energies)**2)).sum(axis = -1)/cell.size()
    
    data += data_baseline
    data *= units
    data_baseline *= units
    
    if orientation == "portrait":
        
        if use_fill:
            plot = axes.fill_betweenx(energies, data, data_baseline, **kwargs)
        else:
            plot = axes.plot(data,energies,**kwargs)
            
        if "Fermi" in cell.meta and show_fermi:
            axes.axhline(y = cell.meta["Fermi"]/units,color='r')
                    
        axes.set_ylim(energy_range)

        if not units_name is None:
            axes.set_xlabel('Density, bands per {}'.format(units_name))
            axes.set_ylabel('Energy, {}'.format(units_name))
            
        else:
            axes.set_xlabel('Density')
            axes.set_ylabel('Energy')

    elif orientation == "landscape":
        
        if use_fill:
            plot = axes.fill_between(energies, data, data_baseline, **kwargs)
        else:
            plot = axes.plot(energies,data,**kwargs)

        if "Fermi" in cell.meta and show_fermi:
            axes.axvline(x = cell.meta["Fermi"]/units,color='r')
            
        axes.set_xlim(energy_range)
        
        if not units_name is None:
            axes.set_ylabel('Density, bands per {}'.format(units_name))
            axes.set_xlabel('Energy, {}'.format(units_name))
            
        else:
            axes.set_ylabel('Density')
            axes.set_xlabel('Energy')

    return plot

def matplotlib_scalar(
    grid,
    axes,
    origin,
    plane,
    units = "angstrom",
    units_name = None,
    show_cell = False,
    ppu = None,
    **kwargs
):
    """
    Plots scalar values on the grid using imshow.
    
    Args:
    
        grid (Grid): a 3D grid to be plotted;
        
        axes (matplotlib.axes.Axes): axes to plot on;
        
        origin (array): origin of the 2D slice to be plotted in the
        units of ``grid``;
        
        plane (str, int): the plotting plane: either 'x','y' or 'z' or a
        correspondint int.
        
    Kwargs:
    
        units (str, float): either a field from ``numericalunits``
        package or a float with energy units;
        
        units_name (str): a string used for the units. Used only if the
        ``units`` keyword is a float;
        
        show_cell (bool): if True then projected unit cell boundaries are
        shown on the final image;
        
        ppu (float): points per ``unit`` for the raster image;
    
        The rest of kwargs are passed to ``pyplot.imshow``.
        
    Returns:
    
        A ``matplotlib.image.AxesImage`` plotted.
    """
    
    if not len(grid.coordinates)==3:
        raise TypeError("A {:d}D grid found, required 3D".format(len(grid.coordinates)))
        
    if not len(grid.values.shape)==3:
        raise TypeError("The dimensionality of the 'values' array is {:d}, required 3".format(len(grid.values.shape)))
        
    if isinstance(units, str):
        units_name = units
        units = getattr(numericalunits, units)
        
    origin = grid.transform_to_cartesian(origin)[numpy.newaxis,:]
    
    plane = __xyz2i__(plane)
    otherVectors = list(range(3))
    del otherVectors[plane]
    
    # Build a rotated cartesian basis
    v1 = grid.vectors[otherVectors][0]
    v3 = grid.vectors[plane]
    v2 = numpy.cross(v3,v1)
    basis = Basis((v1,v2,v3))
    basis.vectors /= ((basis.vectors**2).sum(axis = -1)**.5)[:,numpy.newaxis]
    
    # Calculate in-plane coordinates of the grid edges
    edges_inplane = basis.transform_from_cartesian(grid.vertices() - origin)
    mn = edges_inplane.min(axis = 0)
    mx = edges_inplane.max(axis = 0)
    
    if ppu is None:
        l = ((grid.vectors**2).sum(axis = 1)**.5)
        ppu = (numpy.array(grid.values.shape)/l).min()
        
    else:
        ppu /= units

    # In-plane grid size: px, py
    px = round((mx[0]-mn[0])*ppu)
    py = round((mx[1]-mn[1])*ppu)
    
    # In-plane grid spacing: dx, dy
    dx = (mx[0]-mn[0]) / px
    dy = (mx[0]-mn[0]) / py
    
    # Build an inplane grid
    mg = numpy.meshgrid(numpy.linspace(mn[0]+dx/2,mx[0]-dx/2,px), numpy.linspace(mn[1]+dy/2,mx[1]-dy/2,py), (0,), indexing='ij')
    dims = mg[0].shape[:2]
    points_inplane = numpy.concatenate(tuple(i[...,numpy.newaxis] for i in mg), axis = len(mg)).reshape(-1,3)
    
    # Convert to lattice coordinates of the initial grid
    points_cartesian = basis.transform_to_cartesian(points_inplane) + origin
    points_lattice = grid.transform_from_cartesian(points_cartesian)
    
    interpolated = grid.interpolate_to_cell(points_lattice)
    
    image = axes.imshow(numpy.swapaxes(interpolated.values.reshape(*dims),0,1), extent = [
        mn[0]/units,
        mx[0]/units,
        mn[1]/units,
        mx[1]/units,
    ], origin = "lower", **kwargs)

    if show_cell:
        
        edges = basis.transform_from_cartesian(grid.edges() - origin)/units
        for e in edges:
            axes.plot([e[0,0],e[1,0]],[e[0,1],e[1,1]], color = "black")
    
    axes.set_xlim([mn[0]/units,mx[0]/units])
    axes.set_ylim([mn[1]/units,mx[1]/units])
    
    if not units_name is None:
        axes.set_xlabel(units_name)
        axes.set_ylabel(units_name)
    
    return image

def matplotlib2svgwrite(fig, svg, insert, size, **kwargs):
    """
    Saves a matplotlib image to an existing svgwrite object.
    
    Args:
    
        fig (matplotlib.figure.Figure): a figure to save;
        
        svg (svgwrite.Drawing): an svg drawing to save to;
        
        insert (tuple): a tuple of ints defining destination to insert
        a drawing;
        
        size (tuple): size of the inserted image;
        
    Kwargs:
    
        The kwargs are passed to ``fig.savefig`` used to print the plot.
    """
    
    image_bin = StringIO()
    fig.savefig(image_bin, format = "png", **kwargs)
    image_bin.seek(0)
    image_str = "data:image/png;base64,"+base64.b64encode(image_bin.buf)
    
    svg.add(svg.image(image_str,
        insert = insert,
        size = size,
    ))
