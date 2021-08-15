import open3d as o3d


def xyz2pcd(xyz, input_shape='bcn', color='blue'):
    if input_shape == 'bcn':
        xyz = xyz.permute(1, 0)

    pc = o3d.geometry.PointCloud()
    points = xyz.to('cpu').detach().numpy().copy()
    pc.points = o3d.utility.Vector3dVector(points)
    pc.paint_uniform_color(get_color(color))

    return pc


def get_color(color_name):
    if color_name == "yellow":
        return [1, 0.706, 0]
    if color_name == "blue":
        return [0, 0.651, 0.929]
