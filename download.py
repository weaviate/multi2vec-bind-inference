#!/usr/bin/env python3

from ImageBind.models import imagebind_model
import json

print("Downloading ImageBind model...")

# This command will download model to .checkpoints directory
model = imagebind_model.imagebind_huge(pretrained=True)

# Create an info.json file with model basic information
info = {
  "model": model._get_name(),
  "version": model._version,
}
with open("./.checkpoints/info.json", 'w+') as json_file:
  json.dump(info, json_file, indent = 2)

print("Success")
