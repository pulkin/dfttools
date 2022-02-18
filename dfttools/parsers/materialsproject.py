"""
Parsing `Materials Project <http://materialsproject.org/>`_ responses.
"""
import requests

import numericalunits
import numpy

from .generic import AbstractJSONParser
from ..simple import unit_cell
from ..types import CrystalCell

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
                     headers={
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

    def test_error(self):
        """
        Tests whether any error returned. Raises an exception if so.
        """
        if "valid_response" in self.json and not self.json["valid_response"]:
            text = self.json.get("error", "(None)")
            raise ValueError("The response is invalid, message: {}".format(text))

    @unit_cell
    def cells(self, root=None):
        self.test_error()
        if root is None:
            root = self.json

        result = []

        if isinstance(root, list):
            for i in root:
                result += self.cells(root=i)

        elif isinstance(root, dict):
            if "structure" in root and "@class" in root["structure"] and root["structure"]["@class"] == "Structure":
                structure_root = root["structure"]
                vecs = numpy.array(structure_root["lattice"]["matrix"]) * numericalunits.angstrom
                coords = []
                vals = []
                for s in structure_root["sites"]:
                    coords.append(s["abc"])
                    vals.append(s["label"])
                if "material_id" in root:
                    meta = {"materialsproject-id": root["material_id"]}
                else:
                    meta = None
                return [CrystalCell(vecs, coords, vals, meta=meta)]

            else:
                for k, v in root.items():
                    result += self.cells(root=v)

        return result


jsonr = JSONResponse
