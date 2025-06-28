import torch
import numpy as np


def quads2tris(F):
    out = []
    for f in F:
        if len(f) == 3:
            out += [f]
        elif len(f) == 4:
            out += [[f[0], f[1], f[2]],
                    [f[0], f[2], f[3]]]
        else:
            print("This should not happen...")
    return np.array(out, np.int32)

def get_rotation_matrix(angle, axis):
    axis = axis / np.linalg.norm(axis)
    s = np.sin(angle)
    c = np.cos(angle)

    m = np.zeros((4, 4))

    m[0][0] = axis[0] * axis[0] + (1.0 - axis[0] * axis[0]) * c
    m[0][1] = axis[0] * axis[1] * (1.0 - c) - axis[2] * s
    m[0][2] = axis[0] * axis[2] * (1.0 - c) + axis[1] * s
    m[0][3] = 0.0

    m[1][0] = axis[0] * axis[1] * (1.0 - c) + axis[2] * s
    m[1][1] = axis[1] * axis[1] + (1.0 - axis[1] * axis[1]) * c
    m[1][2] = axis[1] * axis[2] * (1.0 - c) - axis[0] * s
    m[1][3] = 0.0

    m[2][0] = axis[0] * axis[2] * (1.0 - c) - axis[1] * s
    m[2][1] = axis[1] * axis[2] * (1.0 - c) + axis[0] * s
    m[2][2] = axis[2] * axis[2] + (1.0 - axis[2] * axis[2]) * c
    m[2][3] = 0.0

    m[3][0] = 0.0
    m[3][1] = 0.0
    m[3][2] = 0.0
    m[3][3] = 1.0

    return m

def get_matrix_world_to_camera(camera_param=None, cam_pos=None, cam_angle=None):
    if camera_param is None:
        cam_x, cam_y, cam_z = cam_pos
        cam_x_angle, cam_y_angle, cam_z_angle = cam_angle
    else:
        cam_x, cam_y, cam_z = camera_param['pos']
        cam_x_angle, cam_y_angle, cam_z_angle = camera_param['angle']
    # get rotation matrix: from world to camera
    matrix1 = get_rotation_matrix(- cam_x_angle, [0, 1, 0])
    matrix2 = get_rotation_matrix(- cam_y_angle - np.pi, [1, 0, 0])
    rotation_matrix = matrix2 @ matrix1

    # get translation matrix: from world to camera
    translation_matrix = np.zeros((4, 4))
    translation_matrix[0][0] = 1
    translation_matrix[1][1] = 1
    translation_matrix[2][2] = 1
    translation_matrix[3][3] = 1
    translation_matrix[0][3] = - cam_x
    translation_matrix[1][3] = - cam_y
    translation_matrix[2][3] = - cam_z

    return rotation_matrix @ translation_matrix

def intrinsic_from_fov(height, width, fov=90):
    """
    Basic Pinhole Camera Model
    intrinsic params from fov and sensor width and height in pixels
    Returns:
        K:      [4, 4]
    """
    px, py = (width / 2, height / 2)
    hfov = fov / 360. * 2. * np.pi
    fx = width / (2. * np.tan(hfov / 2.))

    vfov = 2. * np.arctan(np.tan(hfov / 2) * height / width)
    fy = height / (2. * np.tan(vfov / 2.))

    return np.array([[fx, 0, px, 0.],
                     [0, fy, py, 0.],
                     [0, 0, 1., 0.],
                     [0., 0., 0., 1.]])

def get_world_coords(rgb, depth, matrix_world_to_camera=None, env=None):
    height, width = depth.shape[:2]
    K = intrinsic_from_fov(height, width, 45)  # the fov is 90 degrees

    # Apply back-projection: K_inv @ pixels * depth
    u0 = K[0, 2]
    v0 = K[1, 2]
    fx = K[0, 0]
    fy = K[1, 1]

    x = np.linspace(0, width - 1, width).astype(np.float)
    y = np.linspace(0, height - 1, height).astype(np.float)
    u, v = np.meshgrid(x, y)
    one = np.ones((height, width, 1))
    x = (u - u0) * depth / fx
    y = (v - v0) * depth / fy
    z = depth
    cam_coords = np.dstack([x, y, z, one])
    if matrix_world_to_camera is None:
        matrix_world_to_camera = get_matrix_world_to_camera(
            env.camera_params[env.camera_name]['pos'], env.camera_params[env.camera_name]['angle'])
    # convert the camera coordinate back to the world coordinate using the rotation and translation matrix
    cam_coords = cam_coords.reshape((-1, 4)).transpose()  # 4 x (height x width)
    world_coords = np.linalg.inv(matrix_world_to_camera) @ cam_coords  # 4 x (height x width)
    world_coords = world_coords.transpose().reshape((height, width, 4))

    return world_coords

def get_pointcloud(depth, cam_mat):
    world_coordinates = get_world_coords(depth, depth, matrix_world_to_camera=cam_mat)
    world_coords = world_coordinates[:, :, :3].reshape((-1, 3))
    pointcloud_positions = world_coords[depth.flatten() > 0]
    return pointcloud_positions

def process_any_cloth(rgb, depth, matrix_world_to_camera,
                      input_type='depth',
                      coords=None,
                      cloth_id=None,
                      real_world=False,
                      normalize_func=None,
                      padding=0,
                      cloth_type="Trousers",
                      ):
    """
    Prepare the input for model inference and privileged information for evaluation

    Returns:
        depth
        pc_sim
        pc_nocs
        cloth_sim_verts
        cloth_nocs_verts
        cloth_faces_tri
        img_nocs
        img_pc
        wnf

        Maybe?
        visibility
    """
    rgb = torch.from_numpy(rgb.copy()).permute(2, 0, 1)
    if real_world and normalize_func is not None:
        depth = normalize_func(depth)
        
    if padding > 0:
        new_depth = np.pad(depth, padding)
        depth = cv2.resize(new_depth, (200, 200), interpolation=cv2.INTER_NEAREST)

    pc_sim = get_pointcloud(depth, matrix_world_to_camera=matrix_world_to_camera)
    scale = (200 + 2 * padding) / 200
    if input_type == 'pc':
        all_idxs = np.arange(len(pc_sim))
        selected_idxs = np.random.choice(all_idxs, size=6000, replace=False)
        # aabb = gdloader.cloth_cano_aabb
        pc_sim_s = pc_sim[selected_idxs].astype(np.float32)
        # cloth_sim_aabb = aabb.reshape((1,) + aabb.shape)
        depth_s = cv2.resize(depth, (200, 200), interpolation=cv2.INTER_NEAREST)
        depth_s = torch.tensor(depth_s, dtype=torch.float32)
        data = {
            'rgb': rgb,
            'x': np.copy(pc_sim_s),
            'depth': depth_s.unsqueeze(0),
            'pos': pc_sim_s,
            # 'cloth_sim_aabb': cloth_sim_aab
        }
    else:
        depth_s = cv2.resize(depth, (200, 200), interpolation=cv2.INTER_NEAREST)
        T_vec_pred = None
        ds_pc = get_pointcloud(depth_s, matrix_world_to_camera).astype(np.float32)
        depth_s = torch.tensor(depth_s, dtype=torch.float32)
        cloth_uv = torch.nonzero(depth_s).long()
        data = {
            'rgb': rgb,
            'depth': depth_s.unsqueeze(0),
            'pos': ds_pc,
            'cloth_uv': cloth_uv,
            'pc_sim': pc_sim,
            'scale': scale,
            'T_vec_pred': T_vec_pred
        }

    if not real_world:
        V, F = readOBJ(f"dataset/cloth3d/{cloth_type}/mesh/{cloth_id:04d}.obj")[:2]
        F = quads2tris(F)
        assert coords.shape[0] == V.shape[0], f"mesh mismatch {coords.shape[0]}  {V.shape[0]}"

        data.update({
            # privileged information only for evaluation purpose
            'cloth_sim_verts': coords,
            'cloth_tri': F,
        })
    data_torch = dict(
        (x[0], torch.from_numpy(x[1]) if isinstance(x[1], np.ndarray) else x[1]) for x in data.items())

    return MyData(**data_torch)

def compute_cloth_mesh_for_planning(vv, medor_model, matrix_world_to_camera, finetune_cfg, env,
                                    gt_ds_id, gt_ds_edges, gt_face, env):
    rgbd = env.get_rgbd(show_picker=False)
    rgb = rgbd[:, :, :3]
    depth = rgbd[:, :, 3]
    # unflattened_depth = depth.copy()
    medor_cfg = medor_model.cfg
    positions = env.get_positions() # N*3 poin clouds
    if vv['pos_mode'] == 'medor':
        input_dict = process_any_cloth(rgb,
                                        depth,
                                        matrix_world_to_camera,
                                        input_type=medor_cfg.input_type,
                                        coords=positions,
                                        cloth_id=env.get_current_config()['cloth_id'],
                                        cloth_type=vv['cloth_type'])
        batch_input = Batch.from_data_list([input_dict],
                                            follow_batch=['cloth_tri', 'cloth_nocs_verts'])

        batch_input = batch_input.to(device=medor_model.device)
        results = medor_model.predict_mesh(batch_input,
                                            finetune_cfg=finetune_cfg,
                                            env=env,
                                            get_flat_canon_pose=True,
                                            )[0]

        verts, edges = results['warp_field_ds'], results['mesh_edges_ds'].T
        faces = results['faces_ds']
        canon_verts = results['flat_verts_ds']
        dense_verts = results['warp_field']

        if vv["tt_finetune"]:
            print('use finetuned ')
            verts = results['opt_warp_field_ds']
            dense_verts = results['opt_warp_field']
        verts_vis, _ = get_visibility_by_rendering(torch.tensor(dense_verts).cuda(),
                                                    torch.tensor(results['faces']).cuda().long())
        verts_vis = verts_vis[results['downsample_id']]

    elif vv['pos_mode'] == 'gt':
        verts, edges, faces = positions[gt_ds_id], gt_ds_edges, gt_face
        verts_vis = get_visible(matrix_world_to_camera, positions, depth).squeeze()
        canon_verts = env.canon_poses.copy()

        verts_vis = verts_vis[gt_ds_id]
        canon_verts = [x[[gt_ds_id]] for x in canon_verts]

    print('Predicted mesh contains {} nodes  {} edges'.format(verts.shape[0], edges.shape[1]))

    picker_position, picked_points = env.action_tool._get_pos()[0], [-1, -1]
    data = {
        'verts': verts,
        'verts_vis': verts_vis,
        'picker_position': picker_position,
        'picked_points': picked_points,
        'mesh_edges': edges,
        'model_face': faces,
        'mapped_particle_indices': None,
        'ds_id': gt_ds_id,
        'model_canon_pos': canon_verts,
        'init_pos': env.init_pos,
        "depth": depth
    }

    return data
