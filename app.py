from bind import Bind, BindInput

text_list=["A dog.", "A car", "A bird"]
image_paths=[".assets/dog_image.jpg", ".assets/car_image.jpg", ".assets/bird_image.jpg"]
audio_paths=[".assets/dog_audio.wav", ".assets/car_audio.wav", ".assets/bird_audio.wav"]

input = BindInput(
  texts=text_list,
  images=image_paths,
  audio=audio_paths,
)

b = Bind("cpu")
res = b.vectorize(input)

print(f"res: {res}")
