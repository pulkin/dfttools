"""
Parsing `Materials Project <http://materialsproject.org/>`_ responses.
"""
import json
import requests

import numpy
import numericalunits

from .generic import AbstractJSONParser
from ..simple import unit_cell
from ..types import UnitCell
from . import default_real_space_basis

GATEWAY = "https://www.materialsproject.org/rest/v1/"

def __request__(api_key, request):
    """
    Performs a request and initial checks of the response of Materials
    Project REST API.
    Args:
        api_key (str): Materials Project API key;
        request (str): Materials Project request;
        
    Returns:
        A json object with the response.
    """
    r = requests.get('{}{}'.format(GATEWAY, request),
        headers = {
            "x-api-key": api_key,
        }
    )
    if r.status_code == 404:
        raise RuntimeError("Server responds 404")
        
    try:
        r_json = r.json()
    except:
        raise RuntimeError("Response JSON could not be parsed:\n{}".format(r.text))
        
    return r_json
    
def check_key(api_key):
    """
    Checks the API key. Raises an exception if check failed.
    Args:
        api_key (str): Materials Project API key;
    """
    r_json = __request__(api_key, "api_check")
        
    if not "api_key_valid" in r_json:
        raise RuntimeError("The response does not contain key 'api_key_valid':\n{}".format(str(r_json)))
        
    if not r_json["api_key_valid"]:
        raise RuntimeError("The key is not valid")

def download_structures(api_key, query, skip_key_check=False):
    """
    Downloads structure information for a given query.
    Args:
        api_key (str): Materials Project API key;
        query (str): query string;
    
    Kwargs:
        skip_key_check (bool): skips the key check if set to True;
        
    Returns:
        A parser of the response data.
    """
    if not skip_key_check:
        check_key(api_key)
    r_json = __request__(api_key, 'materials/{}/vasp/structure'.format(query))
    return JSONResponse(r_json)

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
                b = default_real_space_basis(numpy.array(root["lattice"]["matrix"])*numericalunits.angstrom)
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
