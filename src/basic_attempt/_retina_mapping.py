from random import Random

from abrain import Point3D as Point


def random(seed):
    return lambda *_, rng=Random(seed): Point(
        rng.uniform(-1, 1),
        -1 + .1,
        rng.uniform(-1, 1)
    )


def x_aligned():
    return lambda i, j, k, w, h: Point(
        2 * (w * k + i) / (3 * w - 1) - 1,
        -1 + .1,
        2 * j / (h-1) - 1
    )


def y_aligned():
    return lambda i, j, k, w, h: Point(
        2 * i / (w - 1) - 1,
        -1 + (k + 1) * .1,
        2 * j / (h - 1) - 1
    )


def z_aligned():
    return lambda i, j, k, w, h: Point(
        2 * i / (w - 1) - 1,
        -1 + .1,
        2 * (h * k + j) / (3 * h - 1) - 1
    )
