"""
This submodule contains routines presenting data (unit cell) in various
text formats.
"""
from .data import element_number, element_mass
from .util import dump
from .types import Basis

import numpy
import numericalunits

from itertools import chain
from collections.abc import Iterable
from io import StringIO


def __xsf_structure__(cell, tag=None, indent=4):
    indent = " " * indent
    cell_vectors = ((indent + " {:14.10f}" * 3 + "\n") * 3).format(*numpy.reshape(cell.vectors / numericalunits.angstrom, -1))
    cartesian = cell.cartesian / numericalunits.angstrom
    coords = ''.join(
        (indent + '{:>2} {:14.10f} {:14.10f} {:14.10f}\n'.format(cell.values[i], cartesian[i, 0], cartesian[i, 1],
                                                                 cartesian[i, 2]))
        for i in range(cell.size)
    )
    if tag is None:
        tag = ''
    else:
        tag = ' ' + tag
    return (
                'CRYSTAL\nPRIMVEC{}\n' + cell_vectors + 'CONVVEC{}\n' + cell_vectors + 'PRIMCOORD{}\n{:d} 1\n' + coords).format(
        tag, tag, tag, cell.size
    )


def xsf_structure(*cells):
    """
    Generates an `xcrysden <http://www.xcrysden.org>`_ file with the
    structure.

    Args:

        cells (list): unit cells with atomic coordinates;

    Returns:

        A string contating XSF-formatted data.
    """
    if len(cells) == 1:
        return __xsf_structure__(cells[0])

    answer = "ANIMSTEPS %i\n\n" % len(cells)
    for i in range(len(cells)):
        answer += __xsf_structure__(cells[i], tag=str(i + 1))
        if not i == len(cells):
            answer += '\n'
    return answer


def xsf_grid(grid, cell, npl=6):
    """
    Generates an `xcrysden <http://www.xcrysden.org>`_ file with the
    data on the grid.

    Args:

        grid (Grid): data on the grid;

        cell (Cell): structural data;

    Kwargs:

        npl (int): numbers per line in the grid section;

    Returns:

        A string contating XSF-formatted data.
    """
    result = __xsf_structure__(cell)

    result += "BEGIN_BLOCK_DATAGRID_3D\ndfttools.formatter\nDATAGRID_3D_UNKNOWN\n"
    result += " ".join(("{:d}",) * 3).format(*(numpy.array(grid.values.shape) + 1)) + "\n0 0 0\n"
    result += (("     {:14.10f}" * 3 + "\n") * 3).format(*numpy.reshape(grid.vectors / numericalunits.angstrom, -1))

    l = grid.values
    l = numpy.concatenate((l, l[0:1, :, :]), axis=0)
    l = numpy.concatenate((l, l[:, 0:1, :]), axis=1)
    l = numpy.concatenate((l, l[:, :, 0:1]), axis=2)
    l = l.reshape(-1, order='F')
    for i in range(l.shape[0] // npl):
        result += " ".join(("{:e}",) * npl).format(*l[i * npl:(i + 1) * npl]) + "\n"

    rest = l.shape[0] - (i + 1) * npl
    result += " ".join(("{:e}",) * rest).format(*l[i * npl:]) + "\n"
    result += "END_DATAGRID_3D\nEND_BLOCK_DATAGRID_3D\n"

    return result


def __format_fort__(x, quote=True):
    """
    Formats input into a string recognizable by fortran input libraries.
    Args:
        x (bool, str, int, float): a value tp format;
        quote (bool): if True, quotes strings;

    Returns:
        A formatted string.
    """
    if isinstance(x, bool):
        return {True: ".true.", False: ".false."}[x]
    elif isinstance(x, (int, float, str)):
        t = str if isinstance(x, str) else float if isinstance(x, float) else int if isinstance(x, int) else None
        return {int: "{:d}", float: "{:e}", str: "'{}'" if quote else "{}"}[t].format(x)
    else:
        raise ValueError("Unknown input {}".format(x))


def qe_input(cell=None, relax_mask=0, parameters=None, inline_parameters=None,
             pseudopotentials=None, masses=None, indent=4):
    """
    Generates Quantum Espresso input file.
    Args:
        cell (CrystallCell): a unit cell with atomic coordinates;
        relax_mask (array,int): array with triggers for relaxation
        written as additional columns in the input file;
        parameters (dict): parameters for the input file;
        inline_parameters (dict): a dict of inline parameters such as
        ``crystal_b``, etc;
        pseudopotentials (dict): a dict of pseudopotential file names;
        masses (dict): a dict with elements' masses;
        indent (int): indent size in spaces;

    Returns:
        A string with Quantum Espresso input.
    """
    indent = ' ' * indent
    special = ("control", "system", "electrons", "ions", "cell", "inputpp", "plot", "projwfc")

    if parameters is None:
        parameters = {}
    else:
        parameters = {
            "&" + k.upper() if k.lower() in special else k.upper(): v
            for k, v in parameters.items()
        }

    if inline_parameters is None:
        inline_parameters = {}
    else:
        inline_parameters = {
            "&" + k.upper() if k.lower() in special else k.upper(): v
            for k, v in inline_parameters.items()
        }

    if pseudopotentials is None:
        pseudopotentials = {}

    _masses = element_mass.copy()
    if masses is not None:
        _masses.update({k.lower(): v for k, v in masses.items()})

    if cell is not None:

        if isinstance(relax_mask, int):
            relax_mask = numpy.full((cell.size, 3), relax_mask)
        relax_mask = numpy.array(relax_mask, dtype=int)
        if relax_mask.ndim == 1:
            relax_mask = numpy.repeat(relax_mask[:, numpy.newaxis], 3, axis=1)

        if "&SYSTEM" not in parameters:
            parameters["&SYSTEM"] = {}
        parameters["&SYSTEM"].update(dict(
            ibrav=0,
            ntyp=len(cell.species()),
            nat=cell.size,
        ))

        if "&IONS" not in parameters and parameters.get("&CONTROL", {}).get("calculation", None) in (
        'relax', 'md', 'vc-relax', 'vc-md'):
            parameters["&IONS"] = {}

        # Unit cell
        parameters["CELL_PARAMETERS"] = "\n".join((indent + "{:.14e} {:.14e} {:.14e}",) * 3).format(
            *numpy.reshape(cell.vectors / numericalunits.angstrom, -1))
        inline_parameters["CELL_PARAMETERS"] = "angstrom"

        # Atomic coordinates
        parameters["ATOMIC_POSITIONS"] = "\n".join(
            "{indent}{name:>2s} {x:16.14f} {y:16.14f} {z:16.14f} {fx:d} {fy:d} {fz:d}".format(
                indent=indent,
                name=cell.values[i],
                x=cell.coordinates[i, 0],
                y=cell.coordinates[i, 1],
                z=cell.coordinates[i, 2],
                fx=relax_mask[i, 0],
                fy=relax_mask[i, 1],
                fz=relax_mask[i, 2],
            )
            for i in range(cell.size)
        )
        inline_parameters["ATOMIC_POSITIONS"] = "crystal"

        # Pseudopotentials
        parameters["ATOMIC_SPECIES"] = "\n".join(
            "{indent}{name:2s} {mass:.3f} {data:s}".format(
                indent=indent,
                name=s,
                mass=_masses[s.lower()],
                data=pseudopotentials[s],
            ) for s in sorted(cell.species())
        )

    # Ordering of sections
    def qe_order(a):
        order = {
            "&CONTROL": 0,
            "&SYSTEM": 1,
            "&ELECTRONS": 2,
            "&IONS": 3,
            "&CELL": 4,
            "ATOMIC_SPECIES": 5,
            "ATOMIC_POSITIONS": 6,
            "K_POINTS": 7,
            "CELL_PARAMETERS": 8,
            "OCCUPATIONS": 9,
            "CONSTRAINTS": 10,
            "ATOMIC_FORCES": 11,
        }
        return order[a[0]] if a[0] in order else 1000

    # Compose everything
    result = []
    for section, data in sorted(parameters.items(), key=qe_order):

        if section in inline_parameters:
            result.append("{} {}".format(section, inline_parameters[section]))
            del inline_parameters[section]
        else:
            result.append(section)

        if isinstance(data, dict):
            for key, value in sorted(data.items()):
                result.append("{indent}{key} = {value}".format(
                    indent=indent,
                    key=key,
                    value=__format_fort__(value),
                ))

        elif isinstance(data, str):
            result.append(data)

        else:
            raise ValueError("Unknown object to insert (not a dict or str): {}".format(data))

        if section[0] == "&":
            result.append("/")

    if len(inline_parameters) > 0:
        raise ValueError("Keys {} are present in inline_parameters but not in parameters".format(
            ", ".join(inline_parameters.keys())))
    return "\n".join(result)


def wannier90_input(cell=None, kpts=None, kp_grid=None, parameters=None, block_parameters=None,
                    indent=4):
    """
    Wannier90 input file.
    Args:
        cell (CrystallCell): a unit cell with atomic coordinates;
        kpts (array): k-points list (crystal coordinates);
        kp_grid (list, tuple): grid dimensions of k-points;
        parameters (dict): wannier90 options;
        block_parameters (dict): wannier90 block options;
        indent (int): indent size in spaces;

    Returns:
        A string with input file contents.
    """

    if parameters is None:
        parameters = {}

    if block_parameters is None:
        block_parameters = {}

    indent = " " * indent

    if cell is not None:
        block_parameters["atoms_frac"] = tuple(
            "{atom} {x:.10f} {y:.10f} {z:.10f}".format(atom=a, x=x, y=y, z=z)
            for a, (x, y, z) in zip(cell.values, cell.coordinates)
        )
        block_parameters["unit_cell_cart"] = tuple(
            "{x:.10f} {y:.10f} {z:.10f}".format(x=x, y=y, z=z)
            for x, y, z in cell.vectors / numericalunits.angstrom
        )

    if kp_grid is not None:
        parameters["mp_grid"] = "{:d} {:d} {:d}".format(*kp_grid)

    if kpts is not None:
        block_parameters["kpoints"] = tuple(
            "{x:.10f} {y:.10f} {z:.10f}".format(x=x, y=y, z=z)
            for x, y, z in kpts
        )

    block_parameters = {k: v if isinstance(v, str) else "\n".join(v) for k, v in block_parameters.items()}

    return "\n".join(chain(
        ("{} = {}".format(k, __format_fort__(v, quote=False))
         for k, v in sorted(parameters.items(), key=lambda x: x[0])
         ),
        ("begin {k}\n{v}\nend {k}".format(k=k, v="\n".join(
            "{indent}{line}".format(indent=indent, line=i)
            for i in v.split("\n")
        ))
         for k, v in sorted(block_parameters.items(), key=lambda x: x[0])
         ),
    ))


def siesta_input(cell, indent=4):
    """
    Generates Siesta minimal input file with atomic structure.

    Args:

        cell (Cell): input unit cell;

    Kwargs:

        indent (int): size of indent;

    Returns:

        String with Siesta input file contents.
    """
    indent = ' ' * indent
    species = tuple(cell.species().keys())

    section_csl = "\n".join(tuple(
        indent + "{:d} {:d} {}".format(i + 1, element_number[s.lower()], s)
        for i, s in enumerate(species)
    ))

    section_ac = "\n".join(tuple(
        indent + "{:16.14f} {:16.14f} {:16.14f} {:d}".format(
            cell.coordinates[i, 0],
            cell.coordinates[i, 1],
            cell.coordinates[i, 2],
            species.index(cell.values[i]),
        )
        for i in range(cell.size)
    ))

    section_lv = "\n".join(tuple(
        indent + "{:e} {:e} {:e}".format(*v / numericalunits.angstrom)
        for v in cell.vectors
    ))

    return """NumberOfAtoms {anum:d}
NumberOfSpecies {snum:d}

LatticeConstant 1 Ang
%block LatticeVectors
{section_lv}
%endblock LatticeVectors

%block ChemicalSpeciesLabel
{section_csl}
%endblock ChemicalSpeciesLabel

AtomicCoordinatesFormat Fractional
%block AtomicCoordinatesAndAtomicSpecies
{section_ac}
%endblock AtomicCoordinatesAndAtomicSpecies""".format(
        anum=cell.size,
        snum=len(species),
        section_lv=section_lv,
        section_csl=section_csl,
        section_ac=section_ac,
    )


def __format_openmx__(x):
    """
    Formats input into a string recognizable by OpenMX input parser.
    Args:
        x (bool, str, int, float): a value to format;

    Returns:
        A formatted string.
    """
    if isinstance(x, bool):
        return {True: "on", False: "off"}[x]
    elif isinstance(x, (int, float, str)):
        t = str if isinstance(x, str) else float if isinstance(x, float) else int if isinstance(x, int) else None
        return {int: "{:d}", float: "{:e}", str: "{}"}[t].format(x)
    else:
        raise ValueError("Unknown input {}".format(x))


def openmx_input(cell, populations, parameters=None, block_parameters=None,
                 relax_mask=None, l=None, r=None, tolerance=1e-10, indent=4,
                 pseudos=None, bands=None):
    """
    Generates OpenMX minimal input file with atomic structure.
    Args:
        cell (Cell): input unit cell;
        populations (dict): a dict with initial electronic populations data;
        parameters (dict): a dict with parameters;
        block_parameters (dict): a dict with block parameters;
        relax_mask (array,int): array with triggers for relaxation;
        l (Cell): left lead for transport calculations;
        r (Cell): right lead for transport calculations;
        tolerance (float): tolerance for checking whether left-center-right
        unit cells can be stacked;
        indent (int): size of indent;
        pseudos (dict): a collection of pseudopotential/basis strings;
        bands (list): a list of k-points to calculate bands along;

    Returns:

        String with OpenMX input file formatted data.
    """
    if parameters is None:
        parameters = {}

    if block_parameters is None:
        block_parameters = {}

    indent = ' ' * indent
    left = l
    right = r

    if left is not None and right is not None:
        target = left.stack(cell, right, vector=0, tolerance=tolerance)
        frac = False

    elif left is None and right is None:
        target = cell
        frac = True

    else:
        raise ValueError("Only one of 'left' and 'right' unit cells specified")

    if relax_mask is not None:
        relax_mask = numpy.array(relax_mask, dtype=bool).astype(int)
        block_parameters["md.fixed.xyz"] = "\n".join(
            "{indent}{:d} {:d} {:d}".format(*i, indent=indent)
            for i in relax_mask
        )

    c = target.cartesian / numericalunits.angstrom
    v = target.values

    parameters["atoms.number"] = cell.size
    parameters["species.number"] = len(target.species())

    if frac:
        parameters["atoms.unitvectors.unit"] = "Ang"
        block_parameters["atoms.unitvectors"] = "\n".join(tuple(
                "{indent}{:.15e} {:.15e} {:.15e}".format(*v, indent=indent)
                for v in target.vectors / numericalunits.angstrom
            ))

    def __coords__(fr, num, frac=False):
        _c_ = c if not frac else target.coordinates
        return "\n".join(tuple(
            "{indent}{:3d} {:>2s} {:.15e} {:.15e} {:.15e} {}".format(
                i + 1, v[n], _c_[n, 0], _c_[n, 1], _c_[n, 2], populations[v[n]],
                indent=indent,
            ) for i, n in enumerate(range(fr, fr + num))
        ))

    offset = left.size if not frac else 0

    parameters["atoms.speciesandcoordinates.unit"] = "frac" if frac else "ang"
    block_parameters["atoms.speciesandcoordinates"] = __coords__(offset, cell.size, frac=frac)

    if left is not None:
        parameters["leftleadatoms.number"] = left.size
        block_parameters["leftleadatoms.speciesandcoordinates"] = __coords__(0, left.size)

    if right is not None:
        offset += cell.size
        parameters["rightleadatoms.number"] = right.size
        block_parameters["rightleadatoms.speciesandcoordinates"] = __coords__(offset, right.size)

    if pseudos is not None:
        species = sorted(target.species().keys())
        block_parameters["definition.of.atomic.species"] = "\n".join("{indent}{key} {value}".format(
            indent=indent, key=i, value=pseudos[i]) for i in species)

    if bands is not None:
        fstr = "{indent}2 {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} ? ?"
        parameters["band.dispersion"] = True
        parameters["band.nkpath"] = int(numpy.ceil(len(bands) / 2))
        block_parameters["band.kpath"] = list(
            fstr.format(*i, *j, indent=indent)
            for i, j in zip(bands[::2], bands[1::2])
        )
        if len(block_parameters["band.kpath"]) != parameters["band.nkpath"]:
            block_parameters["band.kpath"].append(fstr.format(*bands[-1], *bands[-1], indent=indent))
        block_parameters["band.kpath"] = "\n".join(block_parameters["band.kpath"])

    result = []
    for k in sorted(chain(parameters.keys(), block_parameters.keys())):
        if k in parameters:
            result.append("{key} {value}".format(key=k, value=__format_openmx__(parameters[k])))
        else:
            result.append("\n<{key}\n{value}\n{key}>\n".format(key=k, value=block_parameters[k]))

    return "\n".join(result)


def pyscf_cell(cell, **kwargs):
    """
    Constructs a unit cell object in pyscf.

    Args:

        cell (Cell): a unit cell object to convert from;

    Kwargs are passed to pyscf.pbc.gto.M.

    Returns:

        A Pyscf Cell object.
    """
    try:
        from pyscf.pbc.gto import Cell
    except ImportError:
        raise

    if "unit" not in kwargs:
        kwargs["unit"] = "Angstrom"

    if kwargs["unit"] == "Angstrom":
        geometry = zip(cell.values, cell.cartesian / numericalunits.angstrom)
        vectors = cell.vectors / numericalunits.angstrom

    elif kwargs["unit"] == "Bohr":
        geometry = zip(cell.values, cell.cartesian / numericalunits.aBohr)
        vectors = cell.vectors / numericalunits.aBohr

    if "dimension" not in kwargs:
        kwargs['dimension'] = 3

    kwargs['atom'] = geometry
    kwargs['a'] = vectors
    c = Cell()
    for k, v in kwargs.items():
        setattr(c, k, v)
    c.build()
    return c


def json_structure(cell, destination=None, **kwargs):
    """
    Outputs the unit cell into JSON string.
    Args:
        cell (Cell, Iterable): a unit cell or
        multiple unit cells to serialize;
        destination (file): optional destination;

    Returns:
        A string with serialized unit cell or None
        if file destination was specified.
    """
    default = dict(indent=2)
    default.update(kwargs)

    if destination is None:
        rtn_str = True
        destination = StringIO()
    else:
        rtn_str = False

    if isinstance(cell, Basis):
        dump(cell.state_dict(), destination, **default)

    elif isinstance(cell, Iterable):
        destination.write("[")
        cnt = False
        for i in cell:
            if cnt:
                destination.write(",")
            dump(i.state_dict(), destination, **default)
            cnt = True
        destination.write("]")

    else:
        raise ValueError("Unknown input: {}".format(cell))

    if rtn_str:
        return destination.getvalue()