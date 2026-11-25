"""Hash the complete FM_P leaf catalog independently in FlamaPy."""
from hashlib import sha256
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True


def catalog(path):
    from flamapy.metamodels.fm_metamodel.transformations.uvl_reader import UVLReader
    from flamapy.metamodels.pysat_metamodel.transformations.fm_to_pysat import FmToPysat
    from flamapy.metamodels.pysat_metamodel.operations.pysat_configurations import PySATConfigurations

    fm = UVLReader(str(path)).transform()
    leaves = {feature.name for feature in fm.get_features() if not feature.relations}
    configs = PySATConfigurations().execute(FmToPysat(fm).transform()).get_result()
    rows = sorted(tuple(sorted(getattr(node, "name", str(node))
                               for node, selected in config.elements.items()
                               if selected and getattr(node, "name", str(node)) in leaves))
                  for config in configs)
    if len(leaves) != 25 or len(fm.ctcs) != 5 or len(rows) != 46368 or len(set(rows)) != len(rows):
        raise ValueError("FM_P SAT catalog differs from the frozen model.")
    payload = "".join("|".join(row) + "\n" for row in rows).encode("ascii")
    return {"configurations": len(rows), "leaf_catalog_sha256": sha256(payload).hexdigest(),
            "uvl_sha256": sha256(path.read_bytes()).hexdigest()}


if __name__ == "__main__":
    print(json.dumps(catalog(Path(sys.argv[1]))))
