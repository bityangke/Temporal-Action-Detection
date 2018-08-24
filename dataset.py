import os
import os.path

import numpy as np
import pandas as pd
import torchvision
from PIL import Image
from torch.utils import data as data
import torch
from ops.fuser import Fuser
from ops.transforms import GroupCenterCrop, ToTorchFormatTensor, GroupRandomCrop, GroupScale, Stack


class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        return int(self._data[1])

    @property
    def label(self):
        return int(self._data[2])

class UnetVideoDataSet(data.Dataset):
    def __init__(self, rgb_list_file, flow_list_file, clip_num=7,
                 rgb_tmpl='image_{:06d}.jpg', flow_tmpl='flow_{}_{:06d}.jpg', segment_length=16,
                 is_thumos14_test_folder=False, transform=None, verbose=False, test=False):
        self.iftest = test
        self.clip_num = clip_num
        self.rgb_tmpl = rgb_tmpl
        self.flow_tmpl = flow_tmpl
        self.transform = transform
        self.segment_length = segment_length
        self.thumos14_test_flag = is_thumos14_test_folder
        self.segment_num = 0
        self.rgb_video_list = [VideoRecord(x.strip().split(' ')) for x in open(rgb_list_file)]
        self.flow_video_list = [VideoRecord(x.strip().split(' ')) for x in open(flow_list_file)]
        self.verbose = verbose

    def __getitem__(self, index):
        rgb_record = self.rgb_video_list[index]
        flow_record = self.flow_video_list[index]
        self.segment_num = flow_record.num_frames // self.segment_length
        rgb_rst = []
        flow_rst = []
        seg_num_in_clip = self.segment_num//self.clip_num
        for i in range(self.clip_num):
            seg_index = i*seg_num_in_clip + np.random.randint(0, seg_num_in_clip)
            rgb_process_data, flow_process_data = self._get_segment(seg_index, [rgb_record, flow_record])
            rgb_rst.append(rgb_process_data)
            flow_rst.append(flow_process_data)
        return torch.stack(rgb_rst), torch.stack(flow_rst), rgb_record.path, rgb_record.label-1 if self.thumos14_test_flag else rgb_record.label, self.segment_num

    def _get_segment(self, seg_index, records):
        if seg_index > self.segment_num:
            raise IndexError(
                "Segement Index Out of Range! Input Seg_ind: {}, Maximum possible segment index: {}.".format(seg_index,
                                                                                                             self.segment_num - 1))
        rgb_images = list()
        flow_images = list()
        for i in range(seg_index * self.segment_length, (seg_index + 1) * self.segment_length):
            rgb_frame, flow_frame = self._load_image(records, i + 1)
            rgb_images.extend(rgb_frame)
            flow_images.extend(flow_frame)
        rgb_process_data = self.transform(rgb_images)
        flow_process_data = self.transform(flow_images)
        return rgb_process_data, flow_process_data

    def _load_image(self, records, idx):
        rgb_record, flow_record = records
        rgb = [Image.open(os.path.join(rgb_record.path, self.rgb_tmpl.format(idx))).convert('RGB')]
        x_img = Image.open(os.path.join(flow_record.path, self.flow_tmpl.format('x', idx))).convert('L')
        y_img = Image.open(os.path.join(flow_record.path, self.flow_tmpl.format('y', idx))).convert('L')
        return rgb, [x_img, y_img]

    def __len__(self):
        return len(self.flow_video_list)






class UnetVideoDataSetNoFixedSegment(data.Dataset):
        def __init__(self, rgb_list_file, flow_list_file, clip_num=7,
                     rgb_tmpl='image_{:06d}.jpg', flow_tmpl='flow_{}_{:06d}.jpg', segment_length=16,
                     is_thumos14_test_folder=False, transform=None, verbose=False, test=False):
            self.iftest = test
            self.clip_num = clip_num
            self.rgb_tmpl = rgb_tmpl
            self.flow_tmpl = flow_tmpl
            self.transform = transform
            self.segment_length = segment_length
            self.thumos14_test_flag = is_thumos14_test_folder
            self.segment_num = 0
            self.rgb_video_list = [VideoRecord(x.strip().split(' ')) for x in open(rgb_list_file)]
            self.flow_video_list = [VideoRecord(x.strip().split(' ')) for x in open(flow_list_file)]
            self.verbose = verbose

        def __getitem__(self, index):
            rgb_record = self.rgb_video_list[index]
            flow_record = self.flow_video_list[index]
            clip_range = flow_record.num_frames // self.clip_num
            rgb_rst = []
            flow_rst = []
            for i in range(self.clip_num):
                if clip_range-self.segment_length+1<=0:
                    print(clip_range, flow_record)
                start_frame = i * clip_range + np.random.randint(0, clip_range-self.segment_length+1)
                rgb_process_data, flow_process_data = self._get_segment(start_frame, [rgb_record, flow_record])
                rgb_rst.append(rgb_process_data)
                flow_rst.append(flow_process_data)
            return torch.stack(rgb_rst), torch.stack(
                flow_rst), rgb_record.path, rgb_record.label - 1 if self.thumos14_test_flag else rgb_record.label, self.segment_num

        def _get_segment(self, start_frame, records):
            assert start_frame+self.segment_length<=records[1].num_frames, (start_frame, records[1].num_frames)
            rgb_images = list()
            flow_images = list()
            for i in range(start_frame, start_frame + self.segment_length):
                rgb_frame, flow_frame = self._load_image(records, i + 1)
                rgb_images.extend(rgb_frame)
                flow_images.extend(flow_frame)
            rgb_process_data = self.transform(rgb_images)
            flow_process_data = self.transform(flow_images)
            return rgb_process_data, flow_process_data

        def _load_image(self, records, idx):
            rgb_record, flow_record = records
            rgb = [Image.open(os.path.join(rgb_record.path, self.rgb_tmpl.format(idx))).convert('RGB')]
            x_img = Image.open(os.path.join(flow_record.path, self.flow_tmpl.format('x', idx))).convert('L')
            y_img = Image.open(os.path.join(flow_record.path, self.flow_tmpl.format('y', idx))).convert('L')
            return rgb, [x_img, y_img]

        def __len__(self):
            return len(self.flow_video_list)


# class UnetValVideoDataSet(data.Dataset):
#     def __init__(self, rgb_list_file, flow_list_file, interval=16,
#                  rgb_tmpl='image_{:06d}.jpg', flow_tmpl='flow_{}_{:06d}.jpg', segment_length=16,
#                  transform=None, verbose=False, test=False):
#         self.iftest = test
#         self.interval = interval
#         self.rgb_tmpl = rgb_tmpl
#         self.flow_tmpl = flow_tmpl
#         self.transform = transform
#         self.segment_length = segment_length
#         self.segment_num = 0
#         self.rgb_video_list = [VideoRecord(x.strip().split(' ')) for x in open(rgb_list_file)]
#         self.flow_video_list = [VideoRecord(x.strip().split(' ')) for x in open(flow_list_file)]
#         self.verbose = verbose
#
#
#     def __getitem__(self, index):
#         rgb_record = self.rgb_video_list[index]
#         flow_record = self.flow_video_list[index]
#         self.segment_num = flow_record.num_frames // self.segment_length
#         rgb_rst = []
#         flow_rst = []
#         # seg_num_in_clip = self.segment_num//self.clip_num
#         for i in range(self.segment_num):
#             rgb_process_data, flow_process_data = self._get_segment(i, [rgb_record, flow_record])
#             rgb_rst.append(rgb_process_data)
#             flow_rst.append(flow_process_data)
#         return torch.stack(rgb_rst), torch.stack(flow_rst), rgb_record.path, rgb_record.label, self.segment_num
#
#     def _get_segment(self, seg_index, records):
#         if seg_index > self.segment_num:
#             raise IndexError(
#                 "Segement Index Out of Range! Input Seg_ind: {}, Maximum possible segment index: {}.".format(seg_index,
#                                                                                                              self.segment_num - 1))
#         rgb_images = list()
#         flow_images = list()
#         for i in range(seg_index * self.segment_length, (seg_index + 1) * self.segment_length):
#             rgb_frame, flow_frame = self._load_image(records, i + 1)
#             rgb_images.extend(rgb_frame)
#             flow_images.extend(flow_frame)
#         rgb_process_data = self.transform(rgb_images)
#         flow_process_data = self.transform(flow_images)
#         return rgb_process_data, flow_process_data
#
#     def _load_image(self, records, idx):
#         rgb_record, flow_record = records
#         rgb = [Image.open(os.path.join(rgb_record.path, self.rgb_tmpl.format(idx))).convert('RGB')]
#         x_img = Image.open(os.path.join(flow_record.path, self.flow_tmpl.format('x', idx))).convert('L')
#         y_img = Image.open(os.path.join(flow_record.path, self.flow_tmpl.format('y', idx))).convert('L')
#         return rgb, [x_img, y_img]
#
#     def __len__(self):
#         return len(self.flow_video_list)





class VideoDataSet(data.Dataset):
    def __init__(self, rgb_list_file, flow_list_file,
                 rgb_tmpl='image_{:06d}.jpg', flow_tmpl='flow_{}_{:06d}.jpg', segment_length=16,
                 transform=None, verbose=False, test=False, test_rate=None):
        if test:
            assert test_rate is not None
        if test_rate:
            assert test_rate < 1 and test_rate > 0
        self.iftest = test
        self.test_rate = test_rate
        self.rgb_tmpl = rgb_tmpl
        self.flow_tmpl = flow_tmpl
        self.transform = transform
        self.segment_length = segment_length
        self.segment_num = 0
        self.rgb_video_list = [VideoRecord(x.strip().split(' ')) for x in open(rgb_list_file)]
        self.flow_video_list = [VideoRecord(x.strip().split(' ')) for x in open(flow_list_file)]
        self.lookup = []
        self.verbose = verbose

        for ind, i in enumerate(self.flow_video_list):
            segment_num = i.num_frames // self.segment_length
            self.lookup.extend([(ind, i) for i in range(segment_num)])

    def __getitem__(self, index):
        if self.iftest:
            index = len(self) + index
        video_index, seg_index = self.lookup[index]
        if self.verbose:
            print("Enter getitem, index: {}, video index: {}, seg_index: {}".format(index, video_index, seg_index))
        rgb_record = self.rgb_video_list[video_index]
        flow_record = self.flow_video_list[video_index]
        self.segment_num = flow_record.num_frames // self.segment_length
        rgb_process_data, flow_process_data = self._get_segment(seg_index, [rgb_record, flow_record])
        return rgb_process_data, flow_process_data, rgb_record.path, rgb_record.label, video_index, seg_index, self.segment_num

    def _get_segment(self, seg_index, records):
        if seg_index > self.segment_num:
            raise IndexError(
                "Segement Index Out of Range! Input Seg_ind: {}, Maximum possible segment index: {}.".format(seg_index,
                                                                                                             self.segment_num - 1))
        rgb_images = list()
        flow_images = list()
        for i in range(seg_index * self.segment_length, (seg_index + 1) * self.segment_length):
            rgb_frame, flow_frame = self._load_image(records, i + 1)
            rgb_images.extend(rgb_frame)
            flow_images.extend(flow_frame)
        rgb_process_data = self.transform(rgb_images)
        flow_process_data = self.transform(flow_images)
        return rgb_process_data, flow_process_data

    def _load_image(self, records, idx):
        rgb_record, flow_record = records
        rgb = [Image.open(os.path.join(rgb_record.path, self.rgb_tmpl.format(idx))).convert('RGB')]
        x_img = Image.open(os.path.join(flow_record.path, self.flow_tmpl.format('x', idx))).convert('L')
        y_img = Image.open(os.path.join(flow_record.path, self.flow_tmpl.format('y', idx))).convert('L')
        return rgb, [x_img, y_img]

    def __len__(self):
        if self.iftest and self.test_rate:
            return int(len(self.lookup) * self.test_rate)
        if (not self.iftest) and self.test_rate:
            return len(self.lookup) - int(len(self.lookup) * self.test_rate)
        return len(self.lookup)


class VideoActionnessDataSet(data.Dataset):
    def __init__(self,
                 rgb_list_file,
                 flow_list_file,
                 bg_rgb_file,
                 bg_flow_file,
                 rgb_tmpl='image_{:06d}.jpg',
                 flow_tmpl='flow_{}_{:06d}.jpg',
                 bg_rgb_tmpl='img_{:05d}.jpg',
                 bg_flow_tmpl='flow_{}_{:05d}.jpg',
                 segment_length=16,
                 bg_rate=0.6,
                 transform=None,
                 verbose=False,
                 test=False,
                 test_rate=None):
        if test:
            assert test_rate is not None
        if test_rate:
            assert test_rate < 1 and test_rate > 0
        self.iftest = test
        self.test_rate = test_rate
        self.rgb_tmpl = rgb_tmpl
        self.flow_tmpl = flow_tmpl
        self.bg_rgb_tmpl = bg_rgb_tmpl
        self.bg_flow_tmpl = bg_flow_tmpl
        self.transform = transform
        self.segment_length = segment_length
        self.segment_num = 0
        self.rgb_video_list = [VideoRecord(x.strip().split(' ')) for x in open(rgb_list_file)]
        self.flow_video_list = [VideoRecord(x.strip().split(' ')) for x in open(flow_list_file)]
        self.bg_rgb_video_list = [VideoRecord(x.strip().split(' ')) for x in open(bg_rgb_file)]
        self.bg_flow_video_list = [VideoRecord(x.strip().split(' ')) for x in open(bg_flow_file)]
        self.lookup = []
        self.verbose = verbose

        fgend = 0
        for i in self.flow_video_list:
            fgend += i.num_frames//self.segment_length
        bgend = 0
        for i in self.bg_flow_video_list:
            bgend += i.num_frames//self.segment_length
        fgstart = 0
        bgstart = 0
        bg_proportion = fgend * bg_rate / (bgend * (1 - bg_rate))

        bgend = int(bgend * bg_proportion)

        if self.iftest and self.test_rate:
            fgstart = fgend - int(fgend * self.test_rate)
            bgstart = bgend - int(bgend * self.test_rate)

        if (not self.iftest) and self.test_rate:
            fgend = fgend - int(fgend * self.test_rate)
            bgend = bgend - int(bgend * self.test_rate)

        tmp_lookup = []
        for ind, i in enumerate(self.flow_video_list):
            segment_num = i.num_frames // self.segment_length
            tmp_lookup.extend([(ind, i, True) for i in range(segment_num)])
        self.lookup.extend(tmp_lookup[fgstart:fgend])

        tmp_lookup = []
        for ind, i in enumerate(self.bg_flow_video_list):
            segment_num = i.num_frames // self.segment_length
            tmp_lookup.extend([(ind, i, False) for i in range(segment_num)])
        self.lookup.extend(tmp_lookup[bgstart:bgend])


    def __getitem__(self, index):
        video_index, seg_index, flag = self.lookup[index]
        if self.verbose:
            print("Enter getitem, index: {}, video index: {}, seg_index: {}, this is {}".format(index, video_index,
                                                                                                seg_index,
                                                                                                'action' if flag else 'background'))
        if flag:
            rgb_record = self.rgb_video_list[video_index]
            flow_record = self.flow_video_list[video_index]
        else:
            rgb_record = self.bg_rgb_video_list[video_index]
            flow_record = self.bg_flow_video_list[video_index]
        self.segment_num = flow_record.num_frames // self.segment_length
        rgb_process_data, flow_process_data = self._get_segment(seg_index, [rgb_record, flow_record], flag)
        return rgb_process_data, flow_process_data, rgb_record.path, rgb_record.label if flag else 101, video_index, seg_index, self.segment_num

    def _get_segment(self, seg_index, records, flag):
        if seg_index > self.segment_num:
            raise IndexError(
                "Segement Index Out of Range! Input Seg_ind: {}, Maximum possible segment index: {}.".format(seg_index,
                                                                                                             self.segment_num - 1))
        rgb_images = list()
        flow_images = list()
        for i in range(seg_index * self.segment_length, (seg_index + 1) * self.segment_length):
            rgb_frame, flow_frame = self._load_image(records, i + 1, flag)
            rgb_images.extend(rgb_frame)
            flow_images.extend(flow_frame)
        rgb_process_data = self.transform(rgb_images)
        flow_process_data = self.transform(flow_images)
        return rgb_process_data, flow_process_data

    def _load_image(self, records, idx, flag):
        rgb_record, flow_record = records
        if flag:
            rgb_tmpl = self.rgb_tmpl
            flow_tmpl = self.flow_tmpl
        else:
            rgb_tmpl = self.bg_rgb_tmpl
            flow_tmpl = self.bg_flow_tmpl
        rgb = [Image.open(os.path.join(rgb_record.path, rgb_tmpl.format(idx))).convert('RGB')]
        x_img = Image.open(os.path.join(flow_record.path, flow_tmpl.format('x', idx))).convert('L')
        y_img = Image.open(os.path.join(flow_record.path, flow_tmpl.format('y', idx))).convert('L')
        return rgb, [x_img, y_img]

    def __len__(self):
        return len(self.lookup)


class FeatureDataset(data.Dataset):
    def __init__(self, list_file, fuser=Fuser('none'), segment_length=16, is_thumos14_test_folder=False):
        super(FeatureDataset, self).__init__()
        self.segment_length = segment_length
        self.fuser = fuser
        self.video_list = pd.read_csv(list_file)
        self.special_flag = is_thumos14_test_folder
        self.ant_flag = 'annotation' in self.video_list.keys()

    def __getitem__(self, index):
        """
        :param index:
        :return: (1)np.ndarray, (2)label, (3)number of segments.
        """
        data = np.load(self.video_list['feature_path'].iloc[index])
        label = self.video_list['label'].iloc[index]
        path = self.video_list['video_path'].iloc[index]
        ant = None
        if self.ant_flag:
            ant = self.video_list['annotation'].iloc[index]
        if self.special_flag:
            label -= 1
        return {
            'data': self.fuser(data),
            'label': label,
            'video_length': data.shape[0] * self.segment_length,
            'video_path': path,
            'annotation': ant
        }

    def __len__(self):
        return len(self.video_list)


def build_video_dataset(dataset, train, test_rate=None, unet=False, unet_clip_num=7):
    assert dataset in ['thumos_validation', 'thumos_test', 'ucf', 'anet', 'background'], dataset
    assert isinstance(train, bool)

    if dataset == 'ucf':
        rgb_list_file = 'ucf_listrgb.txt'
        flow_list_file = 'ucf_listflow.txt'
        rgb_tmpl = 'image_{:04d}.jpg'
        flow_tmpl = 'flow_{}_{:06d}.jpg'
        annotation_root = None
    elif dataset == 'thumos_validation':
        annotation_root = "annotation/validation"
        rgb_list_file = 'thumos_val_rgb.txt'
        flow_list_file = 'thumos_val_flow.txt'
        rgb_tmpl = 'image_{:06d}.jpg'
        flow_tmpl = 'flow_{}_{:06d}.jpg'
    elif dataset == 'thumos_test':
        annotation_root = "annotation/test"
        rgb_list_file = 'thumos_test_rgb.txt'
        flow_list_file = 'thumos_test_flow.txt'
        rgb_tmpl = 'image_{:06d}.jpg'
        flow_tmpl = 'flow_{}_{:06d}.jpg'
    elif dataset == 'background':
        annotation_root = None
        rgb_list_file = 'ucf_listrgb.txt'
        flow_list_file = 'ucf_listflow.txt'
        rgb_tmpl = 'image_{:04d}.jpg'
        flow_tmpl = 'flow_{}_{:06d}.jpg'
        bg_rgb_list_file = 'background_rgb.txt'
        bg_flow_list_file = 'background_flow.txt'
        bg_rgb_tmpl = 'img_{:05d}.jpg'
        bg_flow_tmpl = 'flow_{}_{:05d}.jpg'
    else:
        raise ValueError

    if unet:
        assert dataset in ['thumos_validation', 'thumos_test']
        if train:
            randomcropping = torchvision.transforms.Compose([
                GroupScale(256),
                GroupRandomCrop(224),
            ])
            ds = UnetVideoDataSetNoFixedSegment(rgb_list_file, flow_list_file, rgb_tmpl=rgb_tmpl,
                                        flow_tmpl=flow_tmpl,
                                        clip_num=unet_clip_num,
                                        transform=torchvision.transforms.Compose([
                                            randomcropping,
                                            Stack(roll=False),
                                            ToTorchFormatTensor(div=True),
                                        ]))
        else:
            cropping = torchvision.transforms.Compose([
                GroupScale(256),
                GroupCenterCrop(224),
            ])
            ds = UnetVideoDataSetNoFixedSegment(rgb_list_file, flow_list_file, rgb_tmpl=rgb_tmpl,
                                        flow_tmpl=flow_tmpl,
                                        clip_num=unet_clip_num,
                                        transform=torchvision.transforms.Compose([
                                            cropping,
                                            Stack(roll=False),
                                            ToTorchFormatTensor(div=True),
                                        ]), is_thumos14_test_folder=True)
        return {"dataset": ds} if not annotation_root else {'dataset': ds, 'annotation_root': annotation_root}
    if dataset == 'background':
        if train:
            randomcropping = torchvision.transforms.Compose([
                GroupScale(256),
                GroupRandomCrop(224),
            ])
            if test_rate:
                ds = VideoActionnessDataSet(rgb_list_file, flow_list_file, rgb_tmpl=rgb_tmpl,
                                            flow_tmpl=flow_tmpl,
                                            bg_flow_tmpl=bg_flow_tmpl,
                                            bg_rgb_tmpl=bg_rgb_tmpl,
                                            bg_rgb_file=bg_rgb_list_file,
                                            bg_flow_file=bg_flow_list_file,
                                            transform=torchvision.transforms.Compose([
                                                randomcropping,
                                                Stack(roll=False),
                                                ToTorchFormatTensor(div=True),
                                            ]), test=False, test_rate=test_rate)
            else:
                ds = VideoActionnessDataSet(rgb_list_file, flow_list_file, rgb_tmpl=rgb_tmpl,
                                            flow_tmpl=flow_tmpl,
                                            bg_flow_tmpl=bg_flow_tmpl,
                                            bg_rgb_tmpl=bg_rgb_tmpl,
                                            bg_rgb_file=bg_rgb_list_file,
                                            bg_flow_file=bg_flow_list_file,
                                            transform=torchvision.transforms.Compose([
                                                randomcropping,
                                                Stack(roll=False),
                                                ToTorchFormatTensor(div=True),
                                            ]))
        else:
            cropping = torchvision.transforms.Compose([
                GroupScale(256),
                GroupCenterCrop(224),
            ])
            if test_rate:
                ds = VideoActionnessDataSet(rgb_list_file, flow_list_file, rgb_tmpl=rgb_tmpl,
                                            flow_tmpl=flow_tmpl,
                                            bg_flow_tmpl=bg_flow_tmpl,
                                            bg_rgb_tmpl=bg_rgb_tmpl,
                                            bg_rgb_file=bg_rgb_list_file,
                                            bg_flow_file=bg_flow_list_file,
                                            transform=torchvision.transforms.Compose([
                                                cropping,
                                                Stack(roll=False),
                                                ToTorchFormatTensor(div=True),
                                            ]), test=True, test_rate=test_rate)
            else:
                ds = VideoActionnessDataSet(rgb_list_file, flow_list_file, rgb_tmpl=rgb_tmpl,
                                            flow_tmpl=flow_tmpl,
                                            bg_flow_tmpl=bg_flow_tmpl,
                                            bg_rgb_tmpl=bg_rgb_tmpl,
                                            bg_rgb_file=bg_rgb_list_file,
                                            bg_flow_file=bg_flow_list_file,
                                            transform=torchvision.transforms.Compose([
                                                cropping,
                                                Stack(roll=False),
                                                ToTorchFormatTensor(div=True),
                                            ]))
        return {"dataset": ds} if not annotation_root else {'dataset': ds, 'annotation_root': annotation_root}

    if train:
        randomcropping = torchvision.transforms.Compose([
            GroupScale(256),
            GroupRandomCrop(224),
        ])
        if test_rate:
            ds = VideoDataSet(rgb_list_file, flow_list_file, rgb_tmpl=rgb_tmpl,
                              flow_tmpl=flow_tmpl,
                              transform=torchvision.transforms.Compose([
                                  randomcropping,
                                  Stack(roll=False),
                                  ToTorchFormatTensor(div=True),
                              ]), test=False, test_rate=test_rate)
        else:
            ds = VideoDataSet(rgb_list_file, flow_list_file, rgb_tmpl=rgb_tmpl,
                              flow_tmpl=flow_tmpl,
                              transform=torchvision.transforms.Compose([
                                  randomcropping,
                                  Stack(roll=False),
                                  ToTorchFormatTensor(div=True),
                              ]))
    else:
        cropping = torchvision.transforms.Compose([
            GroupScale(256),
            GroupCenterCrop(224),
        ])
        if test_rate:
            ds = VideoDataSet(rgb_list_file, flow_list_file, rgb_tmpl=rgb_tmpl,
                              flow_tmpl=flow_tmpl,
                              transform=torchvision.transforms.Compose([
                                  cropping,
                                  Stack(roll=False),
                                  ToTorchFormatTensor(div=True),
                              ]), test=True, test_rate=test_rate)
        else:
            ds = VideoDataSet(rgb_list_file, flow_list_file, rgb_tmpl=rgb_tmpl,
                              flow_tmpl=flow_tmpl,
                              transform=torchvision.transforms.Compose([
                                  cropping,
                                  Stack(roll=False),
                                  ToTorchFormatTensor(div=True),
                              ]))
    return {"dataset": ds} if not annotation_root else {'dataset': ds, 'annotation_root': annotation_root}
