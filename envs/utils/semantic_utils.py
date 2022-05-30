import numpy as np
from torchvision import transforms
from PIL import Image



def preprocess_obs(args, obs, use_seg, sem_pred):
    res = transforms.Compose(
            [transforms.ToPILImage(),
             transforms.Resize((args.frame_height, args.frame_width),
                               interpolation=Image.NEAREST)])
    # obs = obs.transpose(1, 2, 0)
    rgb = obs[:, :, :3]*255
    depth = obs[:, :, 3:4]

    sem_seg_pred = get_sem_pred(sem_pred,
        rgb.astype(np.uint8), use_seg=use_seg)
    depth = preprocess_depth(depth, args.min_depth, args.max_depth)
    # depth = depth[:, :, 0]

    ds = args.env_frame_width // args.frame_width  # Downscaling factor
    if ds != 1:
        rgb = np.asarray(res(rgb.astype(np.uint8)))
        depth = depth[ds // 2::ds, ds // 2::ds]
        sem_seg_pred = sem_seg_pred[ds // 2::ds, ds // 2::ds]

    depth = np.expand_dims(depth, axis=2)
    state = np.concatenate((rgb, depth, sem_seg_pred),
                           axis=2).transpose(2, 0, 1)
    return state


def preprocess_depth(depth, min_d, max_d):
    depth = depth[:, :, 0] * 1

    for i in range(depth.shape[1]):
        depth[:, i][depth[:, i] == 0.] = depth[:, i].max()

    mask2 = depth > 0.99
    depth[mask2] = 0.

    mask1 = depth == 0
    depth[mask1] = 100.0
    depth = min_d * 100.0 + depth * max_d * 100.0
    return depth


def get_sem_pred(sem_pred, rgb, use_seg=True):
    if use_seg:
        semantic_pred, rgb_vis = sem_pred.get_prediction(rgb)
        semantic_pred = semantic_pred.astype(np.float32)
    else:
        semantic_pred = np.zeros((rgb.shape[0], rgb.shape[1], 16))
        rgb_vis = rgb[:, :, ::-1]
    return semantic_pred