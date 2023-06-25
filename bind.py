import torch
from pydantic import BaseModel
import asyncio
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

from ImageBind.models import imagebind_model
from ImageBind.models.imagebind_model import ModalityType
import bind_data as data


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
  lock: Lock
  executor: ThreadPoolExecutor

  def __init__(self, cuda: bool, cuda_core: str) -> None:
    self.lock = Lock()
    self.executor = ThreadPoolExecutor()
    self.device = 'cpu'
    if cuda:
      self.device=cuda_core

    self.model = imagebind_model.imagebind_huge(pretrained=True)
    self.model.eval()
    self.model.to(self.device)

  def _get_embeddings(self, inputs):
    with torch.no_grad():
      try:
        self.lock.acquire()
        embeddings = self.model(inputs)
        return embeddings
      finally:
        self.lock.release()

  def _vectorize(self, input: BindInput) -> BindResult:
    inputs = {}
    if input.texts is not None and len(input.texts) > 0:
      inputs[ModalityType.TEXT] = data.load_and_transform_text(input.texts, self.device)
    if input.images is not None and len(input.images) > 0:
      inputs[ModalityType.VISION] = data.load_and_transform_vision_data(input.images, self.device)
    if input.audio is not None and len(input.audio) > 0:
      inputs[ModalityType.AUDIO] = data.load_and_transform_audio_data(input.audio, self.device)
    if input.depth is not None and len(input.depth) > 0:
      inputs[ModalityType.DEPTH] = data.load_and_transform_depth_data(input.depth, self.device)
    if input.imu is not None and len(input.imu) > 0:
      inputs[ModalityType.IMU] = data.load_and_transform_imu_data(input.imu, self.device)
    if input.thermal is not None and len(input.thermal) > 0:
      inputs[ModalityType.THERMAL] = data.load_and_transform_thermal_data(input.thermal, self.device)

    embeddings = self._get_embeddings(inputs)

    text_vectors = embeddings.get(ModalityType.TEXT).tolist() if embeddings.get(ModalityType.TEXT) is not None else []
    image_vectors = embeddings.get(ModalityType.VISION).tolist() if embeddings.get(ModalityType.VISION) is not None else []
    audio_vectors = embeddings.get(ModalityType.AUDIO).tolist() if embeddings.get(ModalityType.AUDIO) is not None else []
    depth_vectors = embeddings.get(ModalityType.DEPTH).tolist() if embeddings.get(ModalityType.DEPTH) is not None else []
    imu_vectors = embeddings.get(ModalityType.IMU).tolist() if embeddings.get(ModalityType.IMU) is not None else []
    thermal_vectors = embeddings.get(ModalityType.THERMAL).tolist() if embeddings.get(ModalityType.THERMAL) is not None else []

    video_vectors = []
    if input.video is not None and len(input.video) > 0:
      inputs = {}
      inputs[ModalityType.VISION] = data.load_and_transform_video_data(input.video, self.device)
      embeddings = self._get_embeddings(inputs)
      video_vectors = embeddings.get(ModalityType.VISION).tolist() if embeddings.get(ModalityType.VISION) is not None else []

    return BindResult(
      text_vectors=text_vectors,
      image_vectors=image_vectors,
      audio_vectors=audio_vectors,
      video_vectors=video_vectors,
      depth_vectors=depth_vectors,
      imu_vectors=imu_vectors,
      thermal_vectors=thermal_vectors,
    )

  async def vectorize(self, payload: BindInput) -> BindResult:
    return await asyncio.wrap_future(self.executor.submit(self._vectorize, payload))
