# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
COCO dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
from pathlib import Path

import torch
import torch.utils.data
import torchvision
import numpy as np
from pycocotools import mask as coco_mask
import datasets.transforms as T
import random

class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, transforms, return_masks, image_set, pkl):
        super(CocoDetection, self).__init__(img_folder, ann_file)
        import pickle 
        
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)
        self.retries = 100

        with open(pkl.replace('train',image_set), 'rb') as f:
            self.proposals = pickle.load(f)

        print("hello")

    def __getitem__(self, idx):
        for _num_retries in range(self.retries):
            try:
                img, target = super(CocoDetection, self).__getitem__(idx)
                shape = torch.as_tensor(img.size)
                image_id = self.ids[idx]
                target = {'image_id': image_id, 'annotations': target}

                img, target = self.prepare(img, target)
                #target['orig_size'] = torch.as_tensor([int(img.size[1]), int(img.size[0])]) # h, w
                target['img_labels'] = torch.unique(target['labels']) # wsod
                target['proposals'] = normalize_proposals(self.proposals[image_id], shape) # wsod proposals from dino

                if self._transforms is not None:
                    img, target = self._transforms(img, target)
                    #target['size'] = torch.as_tensor([int(img.shape[2]), int(img.shape[1])]) # h, w
                # targets.keys() = dict_keys(['boxes', 'labels', 'image_id', 'area', 'iscrowd', 'orig_size', 'size', 'img_labels', 'proposals'])
            except:
                idx = random.randint(0, len(self.ids) - 1)
                continue
            return img, target
        else:
            print(f'Failed in CocoDetection after {_num_retries} retries')

def normalize_proposals(proposals, size):
    # divide with width and height for normalization -> make it simple to resize
    proposals = [p/size[0] if i//2==0 else p/size[1] for i, p in enumerate(proposals)]
    
    return torch.cat(proposals).reshape(-1,4)

def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

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

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target


def make_coco_transforms(image_set):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    #scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
    scales = [640]

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=1333),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=1333), 
                ])
            ),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([640], max_size=1333), #original: 800
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def build(image_set, args):
    root = Path(args.coco_path)
    assert root.exists(), f'provided COCO path {root} does not exist'
    mode = 'instances'
    PATHS = {
        "train": (root / "images" / "train2017", root / "annotations" / f'{mode}_train2017.json'),
        "val": (root / "images" / "val2017", root / "annotations" / f'{mode}_val2017.json'),
    }

    img_folder, ann_file = PATHS[image_set]
    print(img_folder)
    dataset = CocoDetection(img_folder, ann_file, transforms=make_coco_transforms(image_set), return_masks=args.masks, image_set=image_set, pkl=args.pkl)
    
    if args.pkl:
        if args.cache: # want to load partial coco dataset
            cache_path = Path('./data/'+image_set+'_proposals.cache')
            if cache_path.is_file(): 
                print(cache_path+' found..')
                dataset = torch.load(cache_path)
            else:
                import pickle

                with open(args.pkl, 'rb') as f:
                    proposals = pickle.load(f)
                
                dataset = [d for d in dataset if d[1]['image_id'] in list(proposals.keys())]
                torch.save(dataset, cache_path)
                print(cache_path+' saved..')
    return dataset
