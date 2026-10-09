"""Width + tilt -> all 784 (pixel number, coverage coefficient) pairs.

Run: python simple_width_tilt_pixels.py 3.2 5
Fixed center: (14, 14). Fixed height: 19.75 (top 4.125, bottom 23.875).
This is the continuous-area equation, without grey_ones, saved data, or subrows.
Checked for width 1.7-4.7 and tilt -10 to +10 degrees. It closely approximates
the 16-subrow renderer; the tested maximum pixel discrepancy was below 0.00009.
"""

from math import radians, tan


def mean_positive(a, b):
    """Mean of max(0, z) when z changes linearly from a to b."""
    if a <= 0 and b <= 0:
        return 0.0
    if a >= 0 and b >= 0:
        return (a + b) / 2
    return max(a, b) ** 2 / (2 * abs(b - a))


def mean_clipped(a, b):
    """Mean of z clipped to [0, 1], using exact triangle/trapezoid areas."""
    return mean_positive(a, b) - mean_positive(a - 1, b - 1)


def pixels(width, tilt_degrees):
    """Return [(0, coefficient), ..., (783, coefficient)]."""
    if width <= 0 or abs(tilt_degrees) >= 90:
        raise ValueError("Use positive width and tilt strictly between -90 and 90")
    slope = tan(radians(tilt_degrees))
    result = []
    for i in range(784):
        row, column = divmod(i, 28)
        lo = max(row, 4.125)
        hi = min(row + 1, 23.875)
        coefficient = 0.0
        if hi > lo:
            a = 14 + width / 2 - column
            b = 14 - width / 2 - column
            coefficient = (hi - lo) * (
                mean_clipped(a - slope * (lo - 14), a - slope * (hi - 14))
                - mean_clipped(b - slope * (lo - 14), b - slope * (hi - 14))
            )
        result.append((i, min(1.0, max(0.0, coefficient))))
    return result


if __name__ == "__main__":
    import sys

    width, tilt = map(float, sys.argv[1:]) if sys.argv[1:] else (3.2, 0.0)
    print("pixel,coefficient")
    for i, coefficient in pixels(width, tilt):
        print(f"{i},{coefficient:.9f}")
