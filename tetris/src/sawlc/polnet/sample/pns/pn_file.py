"""Module for managing cytosolic proteins configuration files"""

import ast
from pathlib import Path

class PnFile:
    """
    For handling cytosolic proteins configuration files
    """

    def __init__(self):
        self.__params = {}

    @property
    def type(self):
        return self.__params.get("MB_TYPE", None)

    def load(self, in_file: Path) -> None:
        """
        Load protein parameters from an input file

        Args:
            in_file (Path): path to the input file with extension .pns
        """
        if not in_file.suffix == ".pns":
            raise ValueError("Input file must have a .pns extension.")
        if not in_file.exists():
            raise FileNotFoundError(f"Protein file {in_file} does not exist.") 
        with open(in_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "#" in line:
                    line = line.split("#")[
                        0
                    ].strip()
                if "=" in line:
                    key, value = [part.strip() for part in line.split("=", 1)]
                    try:
                        self.__params[key] = ast.literal_eval(value)
                    except (ValueError, SyntaxError):
                        self.__params[key] = (
                            value
                        )
        return self.__params.copy()