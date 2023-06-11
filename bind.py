import torch
from pydantic import BaseModel
from ImageBind.models import imagebind_model
from ImageBind.models.imagebind_model import ModalityType
import ImageBind.data as data


class BindInput(BaseModel):
  texts: list = []
  images: list = []
  audio: list = []
  video: list = []
  imu: list = []
  thermal: list = []
  depth: list = []


class BindResult:
  text_vectors: list = []
  image_vectors: list = []
  audio_vectors: list = []
  video_vectors: list = []
  imu_vectors: list = []
  thermal_vectors: list = []
  depth_vectors: list = []

  def __init__(self, text_vectors, image_vectors, audio_vectors, video_vectors, imu_vectors, thermal_vectors, depth_vectors):
    self.text_vectors = text_vectors
    self.image_vectors = image_vectors
    self.audio_vectors = audio_vectors
    self.video_vectors = video_vectors
    self.imu_vectors = imu_vectors
    self.thermal_vectors = thermal_vectors
    self.depth_vectors = depth_vectors


class Bind:
  def __init__(self, device: str) -> None:
    self.device = device
    self.model = imagebind_model.imagebind_huge(pretrained=True)
    self.model.eval()
    self.model.to(self.device)

  def vectorize(self, input: BindInput) -> BindResult:
    inputs = {}
    if input.texts is not None and len(input.texts) > 0:
      inputs[ModalityType.TEXT] = data.load_and_transform_text(input.texts, self.device)
    if input.images is not None and len(input.images) > 0:
      inputs[ModalityType.VISION] = data.load_and_transform_vision_data(input.images, self.device)
    if input.audio is not None and len(input.audio) > 0:
      inputs[ModalityType.AUDIO] = data.load_and_transform_audio_data(input.audio, self.device)
    if input.video is not None and len(input.video) > 0:
      inputs[ModalityType.VISION] = data.load_and_transform_video_data(input.video, self.device)

    with torch.no_grad():
      embeddings = self.model(inputs)

    text_vectors = embeddings.get(ModalityType.TEXT) if embeddings.get(ModalityType.TEXT) is not None else []
    image_vectors = embeddings.get(ModalityType.VISION) if embeddings.get(ModalityType.VISION) is not None else []
    audio_vectors = embeddings.get(ModalityType.AUDIO) if embeddings.get(ModalityType.AUDIO) is not None else []
    video_vectors = embeddings.get(ModalityType.VISION) if embeddings.get(ModalityType.VISION) is not None else []
    depth_vectors = embeddings.get(ModalityType.DEPTH) if embeddings.get(ModalityType.DEPTH) is not None else []
    imu_vectors = embeddings.get(ModalityType.IMU) if embeddings.get(ModalityType.IMU) is not None else []
    thermal_vectors = embeddings.get(ModalityType.THERMAL) if embeddings.get(ModalityType.THERMAL) is not None else []

    print(f"embeddings[0]: {embeddings[ModalityType.TEXT][0]} len: {len(embeddings[ModalityType.TEXT][0])}")
    print(f"embeddings[1]: {embeddings[ModalityType.VISION][1]} len: {len(embeddings[ModalityType.VISION][1])}")

    print(
        "Vision x Text: ",
        torch.softmax(embeddings[ModalityType.VISION] @ embeddings[ModalityType.TEXT].T, dim=-1),
    )
    print(
        "Audio x Text: ",
        torch.softmax(embeddings[ModalityType.AUDIO] @ embeddings[ModalityType.TEXT].T, dim=-1),
    )
    print(
        "Vision x Audio: ",
        torch.softmax(embeddings[ModalityType.VISION] @ embeddings[ModalityType.AUDIO].T, dim=-1),
    )
    return BindResult(
      text_vectors=text_vectors,
      image_vectors=image_vectors,
      audio_vectors=audio_vectors,
      video_vectors=video_vectors,
      depth_vectors=depth_vectors,
      imu_vectors=imu_vectors,
      thermal_vectors=thermal_vectors,
    )