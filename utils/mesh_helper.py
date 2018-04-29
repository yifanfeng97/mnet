

def read_mesh(file_name):
    points = []
    faces = []
    with open(file_name, 'r') as f:
        line = f.readline().strip()
        if line == 'OFF':
            num_verts, num_faces, num_edge = f.readline().split()
            num_verts = int(num_verts)
            num_faces = int(num_faces)
            num_edge = int(num_edge)
        else:
            num_verts, num_faces, num_edge = line[3:].split()
            num_verts = int(num_verts)
            num_faces = int(num_faces)
            num_edge = int(num_edge)

        for idx in range(num_verts):
            line = f.readline()
            point = [float(v) for v in line.split()]
            points.append(point)

        for idx in range(num_faces):
            line = f.readline()
            face = [int(t_f) for t_f in line.split()]
            faces.append(face)
    return num_verts, num_edge, num_faces, points, faces


def write_mesh(filename, num_verts, num_faces, num_edge, points, faces):
    file_content = 'OFF\n%d %d %d\n' % (num_verts, num_faces, num_edge)
    for idx in range(len(points)):
        file_content += '%f %f %f\n' % (points[idx][0], points[idx][1], points[idx][2])
    for idx in range(len(faces)):
        file_content += '%d %d %d %d\n' % (faces[idx][0], faces[idx][1], faces[idx][2], faces[idx][3])

    with open(filename, 'w') as f:
        f.write(file_content)