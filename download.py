#!/usr/bin/env python3

from ImageBind.models import imagebind_model

# This command will download model to .checkpoints directory
imagebind_model.imagebind_huge(pretrained=True)