import argparse
import sys

import cv2
import numpy as np
import torch

parser = argparse.ArgumentParser(description='Add path of the light-weight openpose repo')
parser.add_argument('--package-dir', type=str,
    default='/media/bob-lytton/MyData/repos/torch_pose')

args = parser.parse_args()
print(args.package_dir)
sys.path.append(args.package_dir)
print(sys.path)

from models.with_mobilenet import PoseEstimationWithMobileNet
from modules.keypoints import extract_keypoints, group_keypoints
from modules.load_state import load_state
from modules.pose import Pose, propagate_ids
from val import normalize, pad_width

class ImageReader(object):
    def __init__(self, file_names):
        self.file_names = file_names
        self.max_idx = len(file_names)

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self):
        if self.idx == self.max_idx:
            raise StopIteration
        img = cv2.imread(self.file_names[self.idx], cv2.IMREAD_COLOR)
        if img.size == 0:
            raise IOError('Image {} cannot be read'.format(self.file_names[self.idx]))
        self.idx = self.idx + 1
        return img


class VideoReader(object):
    def __init__(self, file_name):
        self.file_name = file_name
        try:  # OpenCV needs int to read from webcam
            self.file_name = int(file_name)
        except ValueError:
            pass

    def __iter__(self):
        self.cap = cv2.VideoCapture(self.file_name)
        if not self.cap.isOpened():
            raise IOError('Video {} cannot be opened'.format(self.file_name))
        return self

    def __next__(self):
        was_read, img = self.cap.read()
        if not was_read:
            raise StopIteration
        return img


def infer_fast(net, img, net_input_height_size, stride, upsample_ratio, cpu,
               pad_value=(0, 0, 0), img_mean=(128, 128, 128), img_scale=1/256):
    height, width, _ = img.shape
    scale = net_input_height_size / height

    scaled_img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    scaled_img = normalize(scaled_img, img_mean, img_scale)
    min_dims = [net_input_height_size, max(scaled_img.shape[1], net_input_height_size)]
    padded_img, pad = pad_width(scaled_img, stride, pad_value, min_dims)

    tensor_img = torch.from_numpy(padded_img).permute(2, 0, 1).unsqueeze(0).float()
    if not cpu:
        tensor_img = tensor_img.cuda()

    stages_output = net(tensor_img)

    stage2_heatmaps = stages_output[-2]
    heatmaps = np.transpose(stage2_heatmaps.squeeze().cpu().data.numpy(), (1, 2, 0))
    heatmaps = cv2.resize(heatmaps, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

    stage2_pafs = stages_output[-1]
    pafs = np.transpose(stage2_pafs.squeeze().cpu().data.numpy(), (1, 2, 0))
    pafs = cv2.resize(pafs, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

    return heatmaps, pafs, scale, pad        

def run(net, img, height_size, cpu, track_ids):
    net = net.eval()
    if not cpu:
        net = net.cuda()

    stride = 8
    upsample_ratio = 4
    num_keypoints = Pose.num_kpts
    previous_poses = []
    kpt_names = Pose.kpt_names
    l_wri = kpt_names.index('l_wri')
    r_wri = kpt_names.index('r_wri')
    left_wrists = []
    right_wrists = []

    orig_img = img.copy()
    heatmaps, pafs, scale, pad = infer_fast(net, img, height_size, stride, upsample_ratio, cpu)

    total_keypoints_num = 0
    all_keypoints_by_type = []
    # gather keypoints
    for kpt_idx in range(num_keypoints):  # 19th for bg
        total_keypoints_num += extract_keypoints(heatmaps[:, :, kpt_idx], all_keypoints_by_type, total_keypoints_num)

    # rescale keypoints
    pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, pafs, demo=True)
    for kpt_id in range(all_keypoints.shape[0]):
        all_keypoints[kpt_id, 0] = (all_keypoints[kpt_id, 0] * stride / upsample_ratio - pad[1]) / scale
        all_keypoints[kpt_id, 1] = (all_keypoints[kpt_id, 1] * stride / upsample_ratio - pad[0]) / scale

    # get all poses position
    current_poses = []
    for n in range(len(pose_entries)):
        if len(pose_entries[n]) == 0:
            continue
        pose_keypoints = np.ones((num_keypoints, 2), dtype=np.int32) * -1
        for kpt_id in range(num_keypoints):
            if pose_entries[n][kpt_id] != -1.0:  # keypoint was found
                pose_keypoints[kpt_id, 0] = int(all_keypoints[int(pose_entries[n][kpt_id]), 0])
                pose_keypoints[kpt_id, 1] = int(all_keypoints[int(pose_entries[n][kpt_id]), 1])
        # entries is confidence
        pose = Pose(pose_keypoints, pose_entries[n][18])
        current_poses.append(pose)
        # paint on 
        pose.draw(img)    # draw lines and circles to show the human skeleton found

    img = cv2.addWeighted(orig_img, 0.6, img, 0.4, 0)
    if track_ids == True:
        propagate_ids(previous_poses, current_poses)
        previous_poses = current_poses
        for pose in current_poses:
            left_wrists.append(pose.keypoints[l_wri])
            right_wrists.append(pose.keypoints[r_wri])
            # print(pose.keypoints[l_wri], pose.keypoints[r_wri]) # found: left wrist and right wrist

    return img, current_poses, left_wrists, right_wrists


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='''Lightweight human pose estimation python demo.
                       This is just for quick results preview.
                       Please, consider c++ demo for the best performance.''')
    parser.add_argument('--checkpoint-path', type=str, help='path to the checkpoint', default='/media/bob-lytton/MyData/repos/torch_pose/checkpoint_iter_370000.pth')
    parser.add_argument('--height-size', type=int, default=128, help='network input layer height size')     # 128 is faster than 256, and 2^n is faster with higher accuracy then others
    parser.add_argument('--video', type=str, default='0', help='path to video file or camera id')
    parser.add_argument('--images', nargs='+', default='', help='path to input image(s)')
    parser.add_argument('--cpu', action='store_true', help='run network inference on cpu')
    parser.add_argument('--track-ids', default=True, help='track poses ids')
    args = parser.parse_args()

    if args.video == '' and args.images == '':
        raise ValueError('Either --video or --image has to be provided')

    net = PoseEstimationWithMobileNet()
    checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
    load_state(net, checkpoint)

    frame_provider = ImageReader(args.images)
    if args.video != '':
        frame_provider = VideoReader(args.video)

    for img in frame_provider:
        out_img, current_poses, left_wrists, right_wrists = run(net, img, args.height_size, args.cpu, args.track_ids)
        for pose in current_poses:
            cv2.rectangle(out_img, (pose.bbox[0], pose.bbox[1]),
                          (pose.bbox[0] + pose.bbox[2], pose.bbox[1] + pose.bbox[3]), (0, 255, 0))
            cv2.putText(out_img, 'id: {}'.format(pose.id), (pose.bbox[0], pose.bbox[1] - 16),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255))
        for l_wri in left_wrists:
            print(l_wri[0], l_wri[1])
            cv2.putText(out_img, 'Left Wrist', (l_wri[0], l_wri[1]), cv2.FONT_HERSHEY_COMPLEX_SMALL, 10, (240, 158, 39))
        for r_wri in right_wrists:
            cv2.putText(out_img, 'Right Wrist', (r_wri[0], r_wri[1]), cv2.FONT_HERSHEY_COMPLEX_SMALL, 10, (240, 158, 39))

        cv2.imshow('Lightweight Human Pose Estimation Python Demo', img)

        key = cv2.waitKey(33)
        if key == 27:  # esc
            break
