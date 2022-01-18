

import open3d as o3d
import torch


def tensor2pc(t):
    points = t.to('cpu').detach().numpy().copy()
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points)

    return pc


if __name__ == '__main__':
    p0s_path = 'log/estimated_pc_result/p0s.pt'
    p1s_path = 'log/estimated_pc_result/p1s.pt'
    p1_ests_path = 'log/estimated_pc_result/p1_ests.pt'
    p0s = torch.load(p0s_path, map_location=torch.device('cpu'))
    p1s = torch.load(p1s_path, map_location=torch.device('cpu'))
    p1_ests = torch.load(p1_ests_path, map_location=torch.device('cpu'))

    # 一つだけテストを行う
    source = tensor2pc(p0s[0][0])
    target = tensor2pc(p1s[0][0])
    est = tensor2pc(p1_ests[0][0])

    source.paint_uniform_color([0, 0.651, 0.929])  # yellow
    target.paint_uniform_color([1, 0.706, 0])  # blue
    est.paint_uniform_color([0, 0.651, 0.929])  # yellow
    q = source + target
    a = est + target

    o3d.io.write_point_cloud('log/pc/q.ply', q)
    o3d.io.write_point_cloud('log/pc/a.ply', a)
