import itertools
import pprint

cases = [
    ("default", "x,y,z,d", "b,a")
]


def _str(lst): return ",".join(item for item in lst if item is not None)


for i, (inputs, outputs) in enumerate(itertools.product(
    itertools.product(
        # Depth of module
        [None, "d"],

        # Parent module type
        [None,    # no information
         "b",     # single int encoded (-1: brick, 0: core, 1: hinge)
         "B,H"],  # binary encoded (1, 1 does not make sense)

        # Position on parent
        [None, "dx,dy"],
        # > if core, dx/dy tell about position on face
        # > if brick, dy = 0 and dx tell face
        # > if hinge, dx = dy = 0

        # How big am I?
    ),
    itertools.product(
        # Angle of child
        ["a"],

        # Module type
        ["b", "B,H"],  # Same as for the input

        # Attachment choice
        [None,
         "f",  # Ordering: queried on core face to let the cppn decide the order
         "dx,dy"],  # Explicitly request a *single* attachment

        [None,
         "M0",
         "M0,M1"] # Morphogenes!! "Chemicals". For gradient.
    )
)):
    depth, parent_module_type, parent_offsets = inputs
    angle, module_type, attachment, morphogens = outputs
    if morphogens is not None:
        inputs = (*inputs, morphogens)
    cases.append((str(i), _str(inputs), _str(outputs)))

pprint.pprint(cases)
