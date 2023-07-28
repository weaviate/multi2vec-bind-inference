import json

class Meta:
  def __init__(self):
    self._config = {}
    with open("./.checkpoints/info.json", 'r') as json_file:
      self._config = json.load(json_file)

  async def get(self):
    return self._config