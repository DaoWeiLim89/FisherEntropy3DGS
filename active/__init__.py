from .rand_selector import RandSelector
from .H_reg import HRegSelector
from .H_Entropy import HEntropySelector

methods_dict = {"rand": RandSelector, "H_reg": HRegSelector, "H_Entropy": HEntropySelector}