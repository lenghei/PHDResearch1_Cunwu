"""UVL validity and reproducible sampling via installed FlamaPy."""
import random

def enumerate_uvl(path):
 from flamapy.metamodels.fm_metamodel.transformations.uvl_reader import UVLReader
 from flamapy.metamodels.pysat_metamodel.transformations.fm_to_pysat import FmToPysat
 from flamapy.metamodels.pysat_metamodel.operations.pysat_configurations import PySATConfigurations
 fm=UVLReader(str(path)).transform();configs=PySATConfigurations().execute(FmToPysat(fm).transform()).get_result()
 return fm,configs

def sample_valid(path,budget,seed):
 fm,configs=enumerate_uvl(path)
 if budget<1 or budget>len(configs):raise ValueError('Invalid sampling budget')
 chosen=random.Random(seed).sample(range(len(configs)),budget)
 return dict(valid_count=len(configs),constraint_count=len(fm.ctcs),seed=seed,selected_indices=chosen,features=[sorted(getattr(k,'name',str(k)) for k,v in c.elements.items() if v) for c in [configs[i] for i in chosen]])
