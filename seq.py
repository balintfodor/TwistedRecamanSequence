import numpy as np
import argparse


def recemans_sequence(n):
    # https://oeis.org/A005132
    a, seq = 0, [0]
    for i in range(1, n):
        if (a - i) < 0 or (a - i) in seq:
            a = a + i
        else:
            a = a - i
        seq.append(a)
    return seq


def interpolate_arcs(a0, a1, start_twist_degree, twist_factor, resolution=10):
    # half between a0 and a1
    u = 0.5 * (a0 + a1)
    # radius of the arc
    r = 0.5 * abs(a1 - a0)
    k = int(2 * r * resolution)
    flipped = a0 > a1
    points = []
    for i in range(k):
        t = k - i if flipped else i

        deg = np.pi * (1.0 - t / float(k))
        # z is along the number axis
        z = r * np.cos(deg) + u
        h = abs(r * np.sin(deg))

        s = float(a1 - a0) / k * i + a0 + start_twist_degree
        x = h * np.cos(s * twist_factor)
        y = h * np.sin(s * twist_factor)
        points.append(np.array([x, y, z]))
    return points


def rotation_matrix_from_vectors(a, b):
    v = np.cross(a, b)
    c = np.dot(a, b)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    R = np.identity(3) + vx + vx.dot(vx) * 1.0 / (1 + c)
    return R


def circle_xy(radius, resolution):
    circle = []
    n = resolution
    r = radius
    for i in range(n):
        d = float(i) / n * 2 * np.pi
        x = r * np.cos(d)
        y = r * np.sin(d)
        circle.append(np.array([x, y, 0]))
    return circle


def tube_along_curve(points, radius, resolution, vertex_index_offset=0):
    circle = circle_xy(radius, resolution)
    n = resolution
    vertex_list = []
    for i in range(1, len(points)):
        v = points[i] - points[i - 1]
        v /= np.linalg.norm(v)
        R = rotation_matrix_from_vectors(v, np.array([0, 0, 1]))
        for c in circle:
            vertex_list.append(points[i - 1] + c.dot(R))

    v = points[-1] - points[-2]
    v /= np.linalg.norm(v)
    R = rotation_matrix_from_vectors(v, np.array([0, 0, 1]))
    for c in circle:
        vertex_list.append(points[-1] + c.dot(R))

    face_list = []
    for i in range(0, len(points) - 1):
        for j in range(n):
            unravel = lambda u, v: (i+u)*n + (j+v)%n + 1 + vertex_index_offset
            a = unravel(0, 0)
            b = unravel(1, 0)
            c = unravel(1, 1)
            d = unravel(0, 1)
            face_list.append((a, c, b))
            face_list.append((a, d, c))
    
    return vertex_list, face_list


def save_obj(filename, vertex_list, face_list):
    with open(filename, 'w') as file:
        for v in vertex_list:
            file.write('v {} {} {}\n'.format(v[0], v[1], v[2]))
        
        for idx in face_list:
            file.write('f {} {} {}\n'.format(idx[0], idx[1], idx[2]))


def main():
    parser = argparse.ArgumentParser(description='Generates an .obj model of an interesting visualization of Receman\'s Sequence.')
    parser.add_argument('-n', default=50, type=int, help='Length of the sequence to be generated.')
    parser.add_argument('--twist_factor', default=0.2, type=float, help='Strength of twist on the model, 0 is no twist.')
    parser.add_argument('--arc_resolution', default=10, type=int)
    parser.add_argument('--tube_radius', default=1.0, type=float )
    parser.add_argument('--tube_resolution', default=8, type=int)
    parser.add_argument('--number_axis', default=True, type=bool)
    parser.add_argument('--number_axis_radius', default=1.0, type=float)
    parser.add_argument('--output', default='a.obj', type=str)
    args = parser.parse_args()

    twist_factor = args.twist_factor
    arc_resolution = args.arc_resolution
    tube_radius = args.tube_radius
    tube_resolution = args.tube_resolution
    filename = args.output
    number_axis_radius = args.number_axis_radius

    seq = recemans_sequence(args.n)
    curve = []
    for i in range(1, len(seq)):
        start_twist_degree = 0 if i % 2 == 0 else np.pi / twist_factor
        curve.extend(interpolate_arcs(seq[i-1], seq[i], start_twist_degree, twist_factor, arc_resolution))

    vertex_list, face_list = tube_along_curve(curve, tube_radius, tube_resolution)
    if args.number_axis:
        a = np.array([0.0, 0.0, min(seq)], dtype=np.float)
        b = np.array([0.0, 0.0, max(seq)], dtype=np.float)
        axis_vertices, axis_faces = tube_along_curve([a, b], number_axis_radius, tube_resolution, len(vertex_list))
        vertex_list.extend(axis_vertices)
        face_list.extend(axis_faces)

    save_obj(filename, vertex_list, face_list)


if __name__ == '__main__':
    main()
