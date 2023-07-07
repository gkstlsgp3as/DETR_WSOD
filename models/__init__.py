# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from .detr import build, build_wsod


def build_model(args):
    if args.wsod:
        return build_wsod(args)
    else:
        return build(args)
