import math

def hsv_to_rgb(h: float, s: float, v: float):
    # h [0..1], s [0..1], v [0..1]
    i = int(h * 6.0)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    i = i % 6
    if i == 0: r,g,b = v, t, p
    elif i == 1: r,g,b = q, v, p
    elif i == 2: r,g,b = p, v, t
    elif i == 3: r,g,b = p, q, v
    elif i == 4: r,g,b = t, p, v
    else: r,g,b = v, p, q
    return int(r*255), int(g*255), int(b*255)
