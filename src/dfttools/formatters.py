"""
This submodule contains routines presenting data (unit cell) in various
text formats.
"""
import json

import numpy
from dfttools.presentation import __elements_name_lookup_table__
from numericalunits import angstrom


def __xsf_structure__(cell, tag=None, indent=4):
    indent = " " * indent
    cell_vectors = ((indent + " {:14.10f}" * 3 + "\n") * 3).format(*numpy.reshape(cell.vectors / angstrom, -1))
    cartesian = cell.cartesian() / angstrom
    coords = ''.join(
        (indent + '{:>2} {:14.10f} {:14.10f} {:14.10f}\n'.format(cell.values[i], cartesian[i, 0], cartesian[i, 1],
                                                                 cartesian[i, 2]))
        for i in range(cell.size())
    )
    if tag is None:
        tag = ''
    else:
        tag = ' ' + tag
    return (
                'CRYSTAL\nPRIMVEC{}\n' + cell_vectors + 'CONVVEC{}\n' + cell_vectors + 'PRIMCOORD{}\n{:d} 1\n' + coords).format(
        tag, tag, tag, cell.size()
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
        
        cell (UnitCell): structural data;
        
    Kwargs:
    
        npl (int): numbers per line in the grid section;
        
    Returns:
    
        A string contating XSF-formatted data.
    """
    result = __xsf_structure__(cell)

    result += "BEGIN_BLOCK_DATAGRID_3D\ndfttools.formatter\nDATAGRID_3D_UNKNOWN\n"
    result += " ".join(("{:d}",) * 3).format(*(numpy.array(grid.values.shape) + 1)) + "\n0 0 0\n"
    result += (("     {:14.10f}" * 3 + "\n") * 3).format(*numpy.reshape(grid.vectors / angstrom, -1))

    l = grid.values
    l = numpy.concatenate((l, l[0:1, :, :]), axis=0)
    l = numpy.concatenate((l, l[:, 0:1, :]), axis=1)
    l = numpy.concatenate((l, l[:, :, 0:1]), axis=2)
    l = l.reshape(-1, order='F')
    for i in range(l.shape[0] / npl):
        result += " ".join(("{:e}",) * npl).format(*l[i * npl:(i + 1) * npl]) + "\n"

    rest = l.shape[0] - (i + 1) * npl
    result += " ".join(("{:e}",) * rest).format(*l[i * npl:]) + "\n"
    result += "END_DATAGRID_3D\nEND_BLOCK_DATAGRID_3D\n"

    return result


def qe_input(cell=None, relax_triggers=0, parameters={}, inline_parameters={}, pseudopotentials={}, indent=4):
    """
    Generates Quantum Espresso input file.
    
    Kwargs:
    
        cell (UnitCell): a unit cell with atomic coordinates;
        
        relax_triggers (array,int): array with triggers for relaxation
        written asadditional columns in the input file;
        
        parameters (dict): parameters for the input file;
        
        inline_parameters (dict): a dict of inline parameters such as
        ``crystal_b``, etc;
        
        pseudopotentials (dict): a dict of pseudopotential file names;
        
        indent (int): size of indent;
        
    Returns:
    
        String contating Quantum Espresso input file contents.
    """
    indent = ' ' * indent

    # Parameters to an upper case
    parameters = dict((k.upper(), parameters[k]) for k in parameters)
    inline_parameters = dict((k.upper(), inline_parameters[k]) for k in inline_parameters)

    if not cell is None:

        # Relax_triggers to array
        if isinstance(relax_triggers, int):
            relax_triggers = ((relax_triggers,) * 3,) * cell.size()
        elif len(relax_triggers) == cell.size() and isinstance(relax_triggers[0], int):
            relax_triggers = [(i, i, i) for i in relax_triggers]

        if not "&SYSTEM" in parameters:
            parameters["&SYSTEM"] = {}

        parameters["&SYSTEM"].update({
            "ibrav": 0,
            "ntyp": len(cell.species()),
            "nat": cell.size(),
        })

        if "&IONS" not in parameters and parameters.get("&CONTROL", {}).get("calculation", None) in (
        'relax', 'md', 'vc-relax', 'vc-md'):
            parameters["&IONS"] = {}

        # Unit cell
        parameters["CELL_PARAMETERS"] = "\n".join((indent + "{:.14e} {:.14e} {:.14e}",) * 3).format(
            *numpy.reshape(cell.vectors / angstrom, -1))
        inline_parameters["CELL_PARAMETERS"] = "angstrom"

        # Atomic coordinates
        parameters["ATOMIC_POSITIONS"] = "\n".join(
            "{indent}{name:>2s} {x:16.14f} {y:16.14f} {z:16.14f} {fx:d} {fy:d} {fz:d}".format(
                indent=indent,
                name=cell.values[i],
                x=cell.coordinates[i, 0],
                y=cell.coordinates[i, 1],
                z=cell.coordinates[i, 2],
                fx=relax_triggers[i][0],
                fy=relax_triggers[i][1],
                fz=relax_triggers[i][2],
            )
            for i in range(cell.values.shape[0])
        )
        inline_parameters["ATOMIC_POSITIONS"] = "crystal"

        # Pseudopotentials
        parameters["ATOMIC_SPECIES"] = "\n".join(
            "{indent}{name:2s} {data:s}".format(
                indent=indent,
                name=s,
                data=pseudopotentials[s],
            ) for s in cell.species()
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
    result = ""
    for section, data in sorted(parameters.items(), key=qe_order):

        result += section

        if section in inline_parameters:
            result += " " + inline_parameters[section]
            del inline_parameters[section]

        result += "\n"

        if isinstance(data, dict):

            for key, value in sorted(data.items()):

                if isinstance(value, bool):
                    value = ".true." if value else ".false."
                elif isinstance(value, int):
                    value = "{:d}".format(value)
                elif isinstance(value, float):
                    value = "{:e}".format(value)
                elif isinstance(value, str):
                    value = "'{}'".format(value)
                else:
                    raise ValueError("Unknown data type {}".format(type(value)))

                result += indent + key + " = " + value + "\n"

        elif isinstance(data, str):

            result += data

        if section[0] == "&":
            result += "/\n"
        else:
            result += "\n"

    if len(inline_parameters) > 0:
        raise ValueError("Keys {} are present in inline_parameters but not in parameters".format(
            ", ".join(inline_parameters.keys())))
    return result


def siesta_input(cell, indent=4):
    """
    Generates Siesta minimal input file with atomic structure.
    
    Args:
    
        cell (UnitCell): input unit cell;
        
    Kwargs:
    
        indent (int): size of indent;

    Returns:
    
        String with Siesta input file contents.
    """
    indent = ' ' * indent
    species = cell.species().keys()

    section_csl = "\n".join(tuple(
        indent + "{:d} {:d} {}".format(i + 1, __elements_name_lookup_table__[s.lower()][0], s)
        for i, s in enumerate(species)
    ))

    section_ac = "\n".join(tuple(
        indent + "{:16.14f} {:16.14f} {:16.14f} {:d}".format(
            cell.coordinates[i, 0],
            cell.coordinates[i, 1],
            cell.coordinates[i, 2],
            species.index(cell.values[i]),
        )
        for i in range(cell.size())
    ))

    section_lv = "\n".join(tuple(
        indent + "{:e} {:e} {:e}".format(*v / angstrom)
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
        anum=cell.size(),
        snum=len(species),
        section_lv=section_lv,
        section_csl=section_csl,
        section_ac=section_ac,
    )


def openmx_input(cell, populations, l=None, r=None, tolerance=1e-10, indent=4):
    """
    Generates OpenMX minimal input file with atomic structure.
    
    Args:
    
        cell (UnitCell): input unit cell;
        
        populations (dict): a dict with initial electronic populations data;
        
    Kwargs:
    
        l (UnitCell): left lead;
        
        r (UnitCell): right lead;
        
        tolerance (float): tolerance for checking whether left-center-right
        unit cells can be stacked;
        
        indent (int): size of indent;

    Returns:
    
        String with OpenMX input file formatted data.
    """
    indent = ' ' * indent
    left = l
    right = r

    if not left is None and not right is None:
        target = left.stack(cell, right, vector='x', tolerance=tolerance)
        frac = False

    elif left is None and right is None:
        target = cell
        frac = True

    else:
        raise ValueError("Only one of 'left' and 'right' unit cells specified")

    c = target.cartesian() / angstrom
    v = target.values

    result = """Atoms.Number {anum:d}
Species.Number {snum:d}
""".format(anum=cell.size(), snum=len(target.species()))

    if frac:
        result += """
Atoms.UnitVectors.Unit Ang
<Atoms.UnitVectors
""" + (
            "\n".join(tuple(
                "{indent}{:.15e} {:.15e} {:.15e}".format(*v, indent=indent)
                for v in target.vectors / angstrom
            ))
        ) + """
Atoms.UnitVectors>
"""

    def __coords__(fr, num, frac=False):
        _c_ = c if not frac else target.coordinates
        return "\n".join(tuple(
            "{indent}{:3d} {:>2s} {:.15e} {:.15e} {:.15e} {}".format(
                i + 1, v[n], _c_[n, 0], _c_[n, 1], _c_[n, 2], populations[v[n]],
                indent=indent,
            ) for i, n in enumerate(range(fr, fr + num))
        ))

    offset = left.size() if not frac else 0

    result += """
Atoms.SpeciesAndCoordinates.Unit {}
<Atoms.SpeciesAndCoordinates
""".format("frac" if frac else "ang") + __coords__(offset, cell.size(), frac=frac) + """
Atoms.SpeciesAndCoordinates>
"""

    if not left is None:
        result += """
LeftLeadAtoms.Number {anum:d}""".format(anum=left.size()) + """
<LeftLeadAtoms.SpeciesAndCoordinates
""" + __coords__(0, left.size()) + """
LeftLeadAtoms.SpeciesAndCoordinates>
"""

    if not right is None:
        offset += cell.size()

        result += """
RightLeadAtoms.Number {anum:d}""".format(anum=right.size()) + """
<RightLeadAtoms.SpeciesAndCoordinates
""" + __coords__(offset, right.size()) + """
RightLeadAtoms.SpeciesAndCoordinates>
"""

    return result


def pyscf_cell(cell, **kwargs):
    """
    Constructs a unit cell object in pyscf.
    
    Args:
    
        cell (UnitCell): a unit cell object to convert from;
        
    Kwargs are passed to pyscf.pbc.gto.M.
    
    Returns:
    
        A Pyscf Cell object.
    """
    from pyscf.pbc.gto import Cell

    if not "unit" in kwargs:
        kwargs["unit"] = "Angstrom"

    if kwargs["unit"] == "Angstrom":
        geometry = zip(cell.values, cell.cartesian() / angstrom)
        vectors = cell.vectors / angstrom

    elif kwargs["unit"] == "Bohr":
        geometry = zip(cell.values, cell.cartesian() / aBohr)
        vectors = cell.vectors / aBohr

    if not "dimension" in kwargs:
        kwargs['dimension'] = 3

    kwargs['atom'] = geometry
    kwargs['a'] = vectors
    c = Cell()
    for k, v in kwargs.items():
        setattr(c, k, v)
    c.build()
    return c


def json_structure(cell):
    """
    Outputs the unit cell into JSON string.
    Args:
        cell (UnitCell): a unit cell to serialize;
        
    Returns:
        A string with serialized unit cell.
    """
    return json.dumps(cell.to_json(), indent=2)
