import torch
from torchvision import transforms
from pytorchvideo import transforms as pv_transforms
from pytorchvideo.data.clip_sampling import ConstantClipsPerVideoSampler
from pytorchvideo.data.encoded_video import EncodedVideo
from torchvision import transforms
from torchvision.transforms._transforms_video import NormalizeVideo
from PIL import Image
import base64
import io
from datetime import datetime
import os
from random import randint

import ImageBind.data as data


def load_and_transform_text(text, device):
  return data.load_and_transform_text(text, device)


def load_and_transform_vision_data(images, device):
  if images is None:
        return None

  image_outputs = []
  for base64_encoded_image_string in images:
      data_transform = transforms.Compose(
          [
              transforms.Resize(
                  224, interpolation=transforms.InterpolationMode.BICUBIC
              ),
              transforms.CenterCrop(224),
              transforms.ToTensor(),
              transforms.Normalize(
                  mean=(0.48145466, 0.4578275, 0.40821073),
                  std=(0.26862954, 0.26130258, 0.27577711),
              ),
          ]
      )
      image_bytes = base64.b64decode(base64_encoded_image_string)
      image = Image.open(io.BytesIO(image_bytes))
      if image.mode != 'RGB':
          image = image.convert('RGB')

      image = data_transform(image).to(device)
      image_outputs.append(image)
  return torch.stack(image_outputs, dim=0)


def load_and_transform_audio_data(base64_encoded_audio_files, device):
  try:
    audio_paths = _save_base64_encoded_files(base64_encoded_audio_files)
    result = data.load_and_transform_audio_data(audio_paths, device)
    return result
  finally:
    _remove_files(audio_paths)


def load_and_transform_video_data(base64_encoded_video_paths, device):
  try:
    video_paths = _save_base64_encoded_files(base64_encoded_video_paths)
    result = _load_and_transform_video_data(video_paths, device)
    return result
  finally:
    _remove_files(video_paths)


def _load_and_transform_video_data(
    video_paths,
    device,
    clip_duration=2,
    clips_per_video=5,
    sample_rate=16000,
):
    if video_paths is None:
        return None

    video_outputs = []
    video_transform = transforms.Compose(
        [
            pv_transforms.ShortSideScale(224),
            NormalizeVideo(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )

    clip_sampler = ConstantClipsPerVideoSampler(
        clip_duration=clip_duration, clips_per_video=clips_per_video
    )
    frame_sampler = pv_transforms.UniformTemporalSubsample(num_samples=clip_duration)

    for video_path in video_paths:
        video = EncodedVideo.from_path(
            video_path,
            decode_audio=False,
            # **{"sample_rate": sample_rate},
        )

        all_clips_timepoints = data.get_clip_timepoints(clip_sampler, video.duration)

        all_video = []
        for clip_timepoints in all_clips_timepoints:
            # Read the clip, get frames
            clip = video.get_clip(clip_timepoints[0], clip_timepoints[1])
            if clip is None:
                raise ValueError("No clip found")
            video_clip = frame_sampler(clip["video"])
            video_clip = video_clip / 255.0  # since this is float, need 0-1

            all_video.append(video_clip)

        all_video = [video_transform(clip) for clip in all_video]
        all_video = data.SpatialCrop(224, num_crops=3)(all_video)

        all_video = torch.stack(all_video, dim=0)
        video_outputs.append(all_video)

    return torch.stack(video_outputs, dim=0).to(device)

def _save_base64_encoded_files(base64_encoded_files):
  file_paths = []
  for base64_file in base64_encoded_files:
    file_name = _generate_file_name()
    base64_file_bytes = base64_file.encode('utf-8')
    with open(file_name, 'wb') as file_to_save:
        decoded_file_data = base64.decodebytes(base64_file_bytes)
        file_to_save.write(decoded_file_data)
        file_paths.append(file_name)
  return file_paths


def _generate_file_name():
  file_name = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%f_%p")
  random_number = randint(1000,9999)
  return f"{file_name}_{random_number}"


def _remove_files(file_paths):
  for file_path in file_paths:
    exists = os.path.exists(file_path)
    if exists:
      os.remove(file_path)
