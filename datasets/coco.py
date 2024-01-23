# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Copyright (c) Institute of Information Processing, Leibniz University Hannover.

"""
dataset (COCO-like) which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
import json
import torch
import torch.utils.data
import torchvision
import h5py
import numpy as np
from PIL import Image
import os

import datasets.transforms as T

class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, transforms):
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self._transforms = transforms
        self.prepare = ConvertCocoToVG()

        #TODO load relationship
        with open('/'.join(ann_file.split('/')[:-1])+'/rel.json', 'r') as f:
            all_rels = json.load(f)
        if 'train' in ann_file:
            self.rel_annotations = all_rels['train']
        elif 'val' in ann_file:
            self.rel_annotations = all_rels['val']
        else:
            self.rel_annotations = all_rels['test']

        self.rel_categories = all_rels['rel_categories']

    def __getitem__(self, idx):
        img, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        rel_target = self.rel_annotations[str(image_id)]

        target = {'image_id': image_id, 'annotations': target, 'rel_annotations': rel_target}

        img, target = self.prepare(img, target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target

class ConvertCocoToVG(object):
    def __init__(self):
        pass

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]

        # TODO add relation gt in the target
        rel_annotations = target['rel_annotations']

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        target["image_id"] = image_id

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])
        # TODO add relation gt in the target
        target['rel_annotations'] = torch.tensor(rel_annotations)

        return image, target

BOX_SCALE = 1024

class CocoDetectionVG(torchvision.datasets.CocoDetection):
    def __init__(self, ann_file, split, img_dir, image_file, transforms):
        super(CocoDetectionVG, self).__init__(img_dir, None)
        self._transforms = transforms
        self.prepare = ConvertCocoToVGH5()

        roi_h5 = h5py.File(ann_file, 'r')

        data_split = roi_h5['split_rel'][:]
        if split =='test':
            split_flag = 2
        elif split == 'val':
            split_flag = 1
        else:
            split_flag = 0
        split_mask = data_split == split_flag
        split_mask &= roi_h5['img_to_first_rel'][:] >= 0
        image_index = np.where(split_mask)[0]

        split_mask = np.zeros_like(data_split).astype(bool)
        split_mask[image_index] = True

        # Get box information
        all_labels = roi_h5['labels'][:, 0]
        all_boxes = roi_h5['boxes_{}'.format(BOX_SCALE)][:]  # cx,cy,w,h
        assert np.all(all_boxes[:, :2] >= 0)  # sanity check
        assert np.all(all_boxes[:, 2:] > 0)  # no empty box

        # convert from xc, yc, w, h to x1, y1, x2, y2
        all_boxes[:, :2] = all_boxes[:, :2] - all_boxes[:, 2:] / 2
        all_boxes[:, 2:] = all_boxes[:, :2] + all_boxes[:, 2:]

        im_to_first_box = roi_h5['img_to_first_box'][split_mask]
        im_to_last_box = roi_h5['img_to_last_box'][split_mask]
        im_to_first_rel = roi_h5['img_to_first_rel'][split_mask]
        im_to_last_rel = roi_h5['img_to_last_rel'][split_mask]

        # load relation labels
        _relations = roi_h5['relationships'][:]
        _relation_predicates = roi_h5['predicates'][:, 0]
        assert (im_to_first_rel.shape[0] == im_to_last_rel.shape[0])
        assert (_relations.shape[0] == _relation_predicates.shape[0])  # sanity check

        # Get everything by image.
        self.boxes = []
        self.gt_classes = []
        self.relationships = []
        for i in range(len(image_index)):
            i_obj_start = im_to_first_box[i]
            i_obj_end = im_to_last_box[i]
            i_rel_start = im_to_first_rel[i]
            i_rel_end = im_to_last_rel[i]

            boxes_i = all_boxes[i_obj_start : i_obj_end + 1, :]
            gt_classes_i = all_labels[i_obj_start : i_obj_end + 1]

            if i_rel_start >= 0:
                predicates = _relation_predicates[i_rel_start : i_rel_end + 1]
                obj_idx = _relations[i_rel_start : i_rel_end + 1] - i_obj_start # range is [0, num_box)
                assert np.all(obj_idx >= 0)
                assert np.all(obj_idx < boxes_i.shape[0])
                rels = np.column_stack((obj_idx, predicates)) # (num_rel, 3), representing sub, obj, and pred
            else:
                rels = np.zeros((0, 3), dtype=np.int32)

            self.boxes.append(boxes_i)
            self.gt_classes.append(gt_classes_i)
            self.relationships.append(rels)

        self.filenames, self.img_info = self.load_image_filenames(img_dir, image_file) # length equals to split_mask
        self.filenames = [self.filenames[i] for i in np.where(split_mask)[0]]
        self.img_info = [self.img_info[i] for i in np.where(split_mask)[0]]

        # get all 'image_id' from img_info
        self.ids = [img['image_id'] for img in self.img_info]

    def __getitem__(self, idx):
        # img, target = super(CocoDetectionVG, self).__getitem__(idx)

        img =  Image.open(self.filenames[idx]).convert("RGB")
        target = {}
        target["boxes"] = torch.as_tensor(self.boxes[idx], dtype=torch.float32)
        target["labels"] = torch.as_tensor(self.gt_classes[idx], dtype=torch.int64)

        image_id = self.img_info[idx]['image_id']
        rel_target = self.relationships[idx]

        target["rel_annotations"] = rel_target
        target["image_id"] = image_id

        img, target = self.prepare(img, target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target

    def load_image_filenames(self, img_dir, image_file):
        """
        Loads the image filenames from visual genome from the JSON file that contains them.
        This matches the preprocessing in scene-graph-TF-release/data_tools/vg_to_imdb.py.
        Parameters:
            image_file: JSON file. Elements contain the param "image_id".
            img_dir: directory where the VisualGenome images are located
        Return: 
            List of filenames corresponding to the good images
        """
        with open(image_file, 'r') as f:
            im_data = json.load(f)

        corrupted_ims = ['1592.jpg', '1722.jpg', '4616.jpg', '4617.jpg']
        fns = []
        img_info = []
        for i, img in enumerate(im_data):
            basename = '{}.jpg'.format(img['image_id'])
            if basename in corrupted_ims:
                continue

            filename = os.path.join(img_dir, basename)
            if os.path.exists(filename):
                fns.append(filename)
                img_info.append(img)
        assert len(fns) == len(img_info)
        # assert len(img_info) == 108073
        return fns, img_info

class ConvertCocoToVGH5(object):
    def __init__(self):
        pass

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        boxes = target["boxes"]
        # important: recover original box from BOX_SCALE
        box = boxes / BOX_SCALE * max(w, h)
        # guard against no boxes via resizing
        boxes = torch.as_tensor(box, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = target["labels"]
        classes = torch.tensor(classes, dtype=torch.int64)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]

        rel_annotations = target['rel_annotations']

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        target["image_id"] = image_id

        # for conversion to coco api
        area = torch.tensor([0 for i in range(len(boxes))])
        iscrowd = torch.tensor([0 for i in range(len(boxes))])

        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])
        # TODO add relation gt in the target
        target['rel_annotations'] = torch.tensor(rel_annotations)

        return image, target
    
def make_coco_transforms(image_set):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=1333),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    #T.RandomSizeCrop(384, 600), # TODO: cropping causes that some boxes are dropped then no tensor in the relation part! What should we do?
                    T.RandomResize(scales, max_size=1333),
                ])
            ),
            normalize])

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([800], max_size=1333),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def build(image_set, args):
    ann_file = args.ann_path
    img_dir = args.img_folder
    image_file = args.vg_image_file

    dataset = CocoDetectionVG(ann_file, image_set, img_dir, image_file, transforms=make_coco_transforms(image_set))
    return dataset


# main
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', default='vg')
    parser.add_argument('--ann_path', default='./data/vg/VG-SGG.h5', type=str)
    parser.add_argument('--img_folder', default='./data/vg/images/', type=str)
    parser.add_argument('--vg_image_file', default='./data/vg/image_data.json', type=str)

    args = parser.parse_args()

    dataset = build('train', args)