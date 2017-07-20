"""
Parsing `Materials Project <http://materialsproject.org/>`_ responses.
"""
#import re
#import warnings
#import math
import json
#import os
#import os.path

import numpy
import numericalunits

from .generic import AbstractJSONParser#, parse, cre_varName, cre_word, cre_nonspace, re_int, cre_int, cre_float, AbstractParser, ParseError
#from .structure import cube
#from .native_openmx import openmx_bands_bands
from ..simple import unit_cell#, band_structure, guess_parser, parse, tag_method
from ..types import UnitCell, Basis

class JSONResponse(AbstractJSONParser):
    """
    A class parsing JSON responses of Materials Project API.
    """
    
    @unit_cell
    def unitCells(self, root = None):
        if root is None:
            root = self.json
        
        result = []
        
        if isinstance(root,list):
            for i in root:
                result += self.unitCells(root = i)
                
        elif isinstance(root, dict):
            if "@class" in root and root["@class"] == "Structure":
                b = Basis(numpy.array(root["lattice"]["matrix"])*numericalunits.angstrom)
                coords = []
                vals = []
                for s in root["sites"]:
                    coords.append(s["abc"])
                    vals.append(s["label"])
                return [UnitCell(b,coords,vals)]
            
            else:
                for k,v in root.items():
                    result += self.unitCells(root = v)
                    
        return result
    
jsonr = JSONResponse
