import itertools
import pprint

cases = [
    (0, ("x,y,z,d", "b,a"))
]


def _str(lst): return ",".join(item for item in lst if item is not None)


for inputs, outputs in itertools.product(
    itertools.product(
        # Angle relative to parent
        [None, "a"],

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

        # Neighborhood
        [None, "N"],
    ),
    itertools.product(
        # Angle of child
        ["a"],  # in quarters of circle (0 -> 0, 1 -> 90, 2 -> 180)

        # child module type
        ["b",  # single int encoded (-1: brick, 0: core, 1: hinge)
         "B,H"],  # binary encoded (1, 1 does not make sense)

        # Attachment choice
        [None,
         "f"],  # Ordering: first queried on every attachment to let the cppn decide the order

        # [None,
        #  "M0",
        #  "M0,M1"] # Morphogenes!! "Chemicals". For gradient.
    )
):
    parent_angle, depth, parent_module_type, parent_offsets, neighborhood = inputs
    child_angle, child_module_type, attachment = outputs
    if parent_module_type is not None and parent_module_type != child_module_type:
        continue

    str_inputs, str_outputs = _str(inputs), _str(outputs)
    if len(str_inputs) == 0 or len(str_outputs) == 0:
        continue

    cases.append((len(cases), (str_inputs, str_outputs)))

pprint.pprint(dict(cases))
print("Also NEAT vs not-NEAT")
print("External parameters:")
pprint.pprint(["Depth-First vs Breadth-First"])
