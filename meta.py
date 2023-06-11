class Meta:
  def __init__(self):
    self._config = {
      'model':  'Meta AI ImageBind',
    }
  async def get(self):
    return self._config