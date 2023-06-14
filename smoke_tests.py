import unittest
import requests
import time
import os
import base64

class SmokeTest(unittest.TestCase):
  def setUp(self):
    self.url = 'http://localhost:8000'

    for i in range(0, 100):
      try:
        res = requests.get(self.url + '/.well-known/ready')
        if res.status_code == 204:
          return
        else:
          raise Exception(
                  "status code is {}".format(res.status_code))
      except Exception as e:
        print("Attempt {}: {}".format(i, e))
        time.sleep(1)

    raise Exception("did not start up")

  def testWellKnownReady(self):
    res = requests.get(self.url + '/.well-known/ready')

    self.assertEqual(res.status_code, 204)

  def testWellKnownLive(self):
    res = requests.get(self.url + '/.well-known/live')

    self.assertEqual(res.status_code, 204)

  def testMeta(self):
    res = requests.get(self.url + '/meta')

    self.assertEqual(res.status_code, 200)
    self.assertIsInstance(res.json(), dict)

  def testVectorizing(self):
    text_list=["A dog.", "A car", "A bird"]
    image_paths=["./ImageBind/.assets/dog_image.jpg", "./ImageBind/.assets/car_image.jpg", "./ImageBind/.assets/bird_image.jpg"]
    audio_paths=["./ImageBind/.assets/dog_audio.wav", "./ImageBind/.assets/car_audio.wav", "./ImageBind/.assets/bird_audio.wav"]
    req_body = {
      'texts': text_list,
      'images': convert_to_base64(image_paths),
      'audio': convert_to_base64(audio_paths)
    }
    res = requests.post(self.url + '/vectorize', json=req_body)
    resBody = res.json()

    self.assertEqual(200, res.status_code)
    self.assertTrue(len(resBody['textVectors']) == 3)
    self.assertTrue(len(resBody['imageVectors']) == 3)
    self.assertTrue(len(resBody['audioVectors']) == 3)

    req_body = {'texts':['This is plane'],'images':['/9j/4AAQSkZJRgABAQEASABIAAD/4QpKRXhpZgAASUkqAAgAAAAGABoBBQABAAAAVgAAABsBBQABAAAAXgAAACgBAwABAAAAAgAAADEBAgANAAAAZgAAADIBAgAUAAAAdAAAAGmHBAABAAAAiAAAAJoAAABIAAAAAQAAAEgAAAABAAAAR0lNUCAyLjEwLjE0AAAyMDIxOjAzOjI1IDE2OjI5OjQ3AAEAAaADAAEAAAABAAAAAAAAAAgAAAEEAAEAAAAAAQAAAQEEAAEAAADXAAAAAgEDAAMAAAAAAQAAAwEDAAEAAAAGAAAABgEDAAEAAAAGAAAAFQEDAAEAAAADAAAAAQIEAAEAAAAGAQAAAgIEAAEAAAA7CQAAAAAAAAgACAAIAP/Y/+AAEEpGSUYAAQEAAAEAAQAA/9sAQwAIBgYHBgUIBwcHCQkICgwUDQwLCwwZEhMPFB0aHx4dGhwcICQuJyAiLCMcHCg3KSwwMTQ0NB8nOT04MjwuMzQy/9sAQwEJCQkMCwwYDQ0YMiEcITIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIy/8AAEQgA1wEAAwEiAAIRAQMRAf/EAB8AAAEFAQEBAQEBAAAAAAAAAAABAgMEBQYHCAkKC//EALUQAAIBAwMCBAMFBQQEAAABfQECAwAEEQUSITFBBhNRYQcicRQygZGhCCNCscEVUtHwJDNicoIJChYXGBkaJSYnKCkqNDU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6g4SFhoeIiYqSk5SVlpeYmZqio6Slpqeoqaqys7S1tre4ubrCw8TFxsfIycrS09TV1tfY2drh4uPk5ebn6Onq8fLz9PX29/j5+v/EAB8BAAMBAQEBAQEBAQEAAAAAAAABAgMEBQYHCAkKC//EALURAAIBAgQEAwQHBQQEAAECdwABAgMRBAUhMQYSQVEHYXETIjKBCBRCkaGxwQkjM1LwFWJy0QoWJDThJfEXGBkaJicoKSo1Njc4OTpDREVGR0hJSlNUVVZXWFlaY2RlZmdoaWpzdHV2d3h5eoKDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uLj5OXm5+jp6vLz9PX29/j5+v/aAAwDAQACEQMRAD8A9/ooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAK+YP+GjvGH/QN0P8A78Tf/Ha+n6+AKAPYP+GjvGH/AEDdD/78Tf8Ax2j/AIaO8Yf9A3Q/+/E3/wAdrx+igD2D/ho7xh/0DdD/AO/E3/x2j/ho7xh/0DdD/wC/E3/x2vH6KAPYP+GjvGH/AEDdD/78Tf8Ax2j/AIaO8Yf9A3Q/+/E3/wAdrx+igD2D/ho7xh/0DdD/AO/E3/x2j/ho7xh/0DdD/wC/E3/x2vH6KAPYP+GjvGH/AEDdD/78Tf8Ax2j/AIaO8Yf9A3Q/+/E3/wAdrx+igD2D/ho7xh/0DdD/AO/E3/x2j/ho7xh/0DdD/wC/E3/x2vH6KAPYP+GjvGH/AEDdD/78Tf8Ax2j/AIaO8Yf9A3Q/+/E3/wAdrx+igD2D/ho7xh/0DdD/AO/E3/x2j/ho7xh/0DdD/wC/E3/x2vH6KAPYP+GjvGH/AEDdD/78Tf8Ax2j/AIaO8Yf9A3Q/+/E3/wAdrx+igD2D/ho7xh/0DdD/AO/E3/x2j/ho7xh/0DdD/wC/E3/x2vH6KAPYP+GjvGH/AEDdD/78Tf8Ax2j/AIaO8Yf9A3Q/+/E3/wAdrx+igD7/AKKKKACiiigAooooAKKKKACvgCvv+vgCgAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKAPv8AooooAKKKKACiiigAooooAK+AK+/6+AKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooA+/wCiiigAooooAKKKKACiiigAr4Ar7/r4AoAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigD7/AKKKKACiiigAooooAKKKKACvgCvv+vgCgAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKAPv8AooooAKKKKACiiigAooooAK+AK+/6+AKACiiigAooooAKKKKACiiigAr0jwt8FPEni7w5aa5YXulR2t1v2JPLIHG12Q5AjI6qe9eb19f/AAS/5JDoX/bx/wClElAHkH/DOPjD/oJaH/3/AJv/AI1R/wAM4+MP+glof/f+b/41X0/RQB8wf8M4+MP+glof/f8Am/8AjVH/AAzj4w/6CWh/9/5v/jVfT9FAHzB/wzj4w/6CWh/9/wCb/wCNUf8ADOPjD/oJaH/3/m/+NV9P0UAfMH/DOPjD/oJaH/3/AJv/AI1R/wAM4+MP+glof/f+b/41X0/RQB8wf8M4+MP+glof/f8Am/8AjVH/AAzj4w/6CWh/9/5v/jVfT9FAHzB/wzj4w/6CWh/9/wCb/wCNUf8ADOPjD/oJaH/3/m/+NV9P0UAFFFFABRRRQAUUUUAFFFFABXwBX3/XwBQAUUUUAFFFFABRRRQAUUUUAFfX/wAEv+SQ6F/28f8ApRJXyBX1/wDBL/kkOhf9vH/pRJQB6BRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAV8AV9/wBfAFABRRRQAUUUUAFFFFABRRRQAV9f/BL/AJJDoX/bx/6USV8gV9f/AAS/5JDoX/bx/wClElAHoFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABXwBX3/XwBQAUUUUAFFFFABRRRQAUUUUAFfX/wS/5JDoX/AG8f+lElfIFfX/wS/wCSQ6F/28f+lElAHoFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABXwBX3/XwBQAUUUUAFFFFABRRRQAUUUUAFfX/AMEv+SQ6F/28f+lElfIFfX/wS/5JDoX/AG8f+lElAHoFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABXwBRRQAUUUUAFFFFABRRRQAUUUUAFfX/wAEv+SQ6F/28f8ApRJRRQB6BRRRQAUUUUAFFFFABRRRQAUUUUAFFFFAH//ZAP/iArBJQ0NfUFJPRklMRQABAQAAAqBsY21zBDAAAG1udHJSR0IgWFlaIAflAAMAGQAPABwAMmFjc3BBUFBMAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAD21gABAAAAANMtbGNtcwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADWRlc2MAAAEgAAAAQGNwcnQAAAFgAAAANnd0cHQAAAGYAAAAFGNoYWQAAAGsAAAALHJYWVoAAAHYAAAAFGJYWVoAAAHsAAAAFGdYWVoAAAIAAAAAFHJUUkMAAAIUAAAAIGdUUkMAAAIUAAAAIGJUUkMAAAIUAAAAIGNocm0AAAI0AAAAJGRtbmQAAAJYAAAAJGRtZGQAAAJ8AAAAJG1sdWMAAAAAAAAAAQAAAAxlblVTAAAAJAAAABwARwBJAE0AUAAgAGIAdQBpAGwAdAAtAGkAbgAgAHMAUgBHAEJtbHVjAAAAAAAAAAEAAAAMZW5VUwAAABoAAAAcAFAAdQBiAGwAaQBjACAARABvAG0AYQBpAG4AAFhZWiAAAAAAAAD21gABAAAAANMtc2YzMgAAAAAAAQxCAAAF3v//8yUAAAeTAAD9kP//+6H///2iAAAD3AAAwG5YWVogAAAAAAAAb6AAADj1AAADkFhZWiAAAAAAAAAknwAAD4QAALbEWFlaIAAAAAAAAGKXAAC3hwAAGNlwYXJhAAAAAAADAAAAAmZmAADypwAADVkAABPQAAAKW2Nocm0AAAAAAAMAAAAAo9cAAFR8AABMzQAAmZoAACZnAAAPXG1sdWMAAAAAAAAAAQAAAAxlblVTAAAACAAAABwARwBJAE0AUG1sdWMAAAAAAAAAAQAAAAxlblVTAAAACAAAABwAcwBSAEcAQv/bAEMAAwICAwICAwMDAwQDAwQFCAUFBAQFCgcHBggMCgwMCwoLCw0OEhANDhEOCwsQFhARExQVFRUMDxcYFhQYEhQVFP/bAEMBAwQEBQQFCQUFCRQNCw0UFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFP/CABEIABUAGQMBEQACEQEDEQH/xAAXAAADAQAAAAAAAAAAAAAAAAAABwgJ/8QAFAEBAAAAAAAAAAAAAAAAAAAAAP/aAAwDAQACEAMQAAABpkSAANUzlHgBVRABpSB//8QAGxAAAQUBAQAAAAAAAAAAAAAABQAEBhc2AhD/2gAIAQEAAQUCIPeBrC6giuoIrqCKWZYey7JP6VNqlTalmWiep8//xAAUEQEAAAAAAAAAAAAAAAAAAAAw/9oACAEDAQE/AU//xAAUEQEAAAAAAAAAAAAAAAAAAAAw/9oACAECAQE/AU//xAAhEAAABQQDAQEAAAAAAAAAAAAAAQIDBQQ0k9IRdLIxEP/aAAgBAQAGPwKpq3CUbbDanVEn7wRci1kMaNxayGNG4tZDGjcTPTe8GKakbNJOPuJaSavnJnwLqPyL0F1H5F6CZ6b3gxDdxn2X7//EABcQAQEBAQAAAAAAAAAAAAAAAAERICH/2gAIAQEAAT8hLs8otSCp2GcWLErbs8qEQWDyu8WJWr//2gAMAwEAAgADAAAAEIABAAJP/8QAFBEBAAAAAAAAAAAAAAAAAAAAMP/aAAgBAwEBPxBP/8QAFBEBAAAAAAAAAAAAAAAAAAAAMP/aAAgBAgEBPxBP/8QAFxABAQEBAAAAAAAAAAAAAAAAAREgMf/aAAgBAQABPxAT69h7QKUBQsqdz06dNin17D2oAKoLLB5vp02bP//Z']}
    res = requests.post(self.url + '/vectorize', json=req_body)
    resBody = res.json()

    self.assertEqual(200, res.status_code)
    self.assertTrue(len(resBody['textVectors']) == 1)
    self.assertTrue(len(resBody['imageVectors']) == 1)

    req_body = {'texts':['This is plane','Boeing 737','Blue plane'],'images':['/9j/4AAQSkZJRgABAQEASABIAAD/4QpKRXhpZgAASUkqAAgAAAAGABoBBQABAAAAVgAAABsBBQABAAAAXgAAACgBAwABAAAAAgAAADEBAgANAAAAZgAAADIBAgAUAAAAdAAAAGmHBAABAAAAiAAAAJoAAABIAAAAAQAAAEgAAAABAAAAR0lNUCAyLjEwLjE0AAAyMDIxOjAzOjI1IDE2OjI5OjQ3AAEAAaADAAEAAAABAAAAAAAAAAgAAAEEAAEAAAAAAQAAAQEEAAEAAADXAAAAAgEDAAMAAAAAAQAAAwEDAAEAAAAGAAAABgEDAAEAAAAGAAAAFQEDAAEAAAADAAAAAQIEAAEAAAAGAQAAAgIEAAEAAAA7CQAAAAAAAAgACAAIAP/Y/+AAEEpGSUYAAQEAAAEAAQAA/9sAQwAIBgYHBgUIBwcHCQkICgwUDQwLCwwZEhMPFB0aHx4dGhwcICQuJyAiLCMcHCg3KSwwMTQ0NB8nOT04MjwuMzQy/9sAQwEJCQkMCwwYDQ0YMiEcITIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIy/8AAEQgA1wEAAwEiAAIRAQMRAf/EAB8AAAEFAQEBAQEBAAAAAAAAAAABAgMEBQYHCAkKC//EALUQAAIBAwMCBAMFBQQEAAABfQECAwAEEQUSITFBBhNRYQcicRQygZGhCCNCscEVUtHwJDNicoIJChYXGBkaJSYnKCkqNDU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6g4SFhoeIiYqSk5SVlpeYmZqio6Slpqeoqaqys7S1tre4ubrCw8TFxsfIycrS09TV1tfY2drh4uPk5ebn6Onq8fLz9PX29/j5+v/EAB8BAAMBAQEBAQEBAQEAAAAAAAABAgMEBQYHCAkKC//EALURAAIBAgQEAwQHBQQEAAECdwABAgMRBAUhMQYSQVEHYXETIjKBCBRCkaGxwQkjM1LwFWJy0QoWJDThJfEXGBkaJicoKSo1Njc4OTpDREVGR0hJSlNUVVZXWFlaY2RlZmdoaWpzdHV2d3h5eoKDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uLj5OXm5+jp6vLz9PX29/j5+v/aAAwDAQACEQMRAD8A9/ooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAK+YP+GjvGH/QN0P8A78Tf/Ha+n6+AKAPYP+GjvGH/AEDdD/78Tf8Ax2j/AIaO8Yf9A3Q/+/E3/wAdrx+igD2D/ho7xh/0DdD/AO/E3/x2j/ho7xh/0DdD/wC/E3/x2vH6KAPYP+GjvGH/AEDdD/78Tf8Ax2j/AIaO8Yf9A3Q/+/E3/wAdrx+igD2D/ho7xh/0DdD/AO/E3/x2j/ho7xh/0DdD/wC/E3/x2vH6KAPYP+GjvGH/AEDdD/78Tf8Ax2j/AIaO8Yf9A3Q/+/E3/wAdrx+igD2D/ho7xh/0DdD/AO/E3/x2j/ho7xh/0DdD/wC/E3/x2vH6KAPYP+GjvGH/AEDdD/78Tf8Ax2j/AIaO8Yf9A3Q/+/E3/wAdrx+igD2D/ho7xh/0DdD/AO/E3/x2j/ho7xh/0DdD/wC/E3/x2vH6KAPYP+GjvGH/AEDdD/78Tf8Ax2j/AIaO8Yf9A3Q/+/E3/wAdrx+igD2D/ho7xh/0DdD/AO/E3/x2j/ho7xh/0DdD/wC/E3/x2vH6KAPYP+GjvGH/AEDdD/78Tf8Ax2j/AIaO8Yf9A3Q/+/E3/wAdrx+igD7/AKKKKACiiigAooooAKKKKACvgCvv+vgCgAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKAPv8AooooAKKKKACiiigAooooAK+AK+/6+AKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooA+/wCiiigAooooAKKKKACiiigAr4Ar7/r4AoAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigD7/AKKKKACiiigAooooAKKKKACvgCvv+vgC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+v/aAAwDAQACEQMRAD8A9/ooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAK+YP+GjvGH/QN0P8A78Tf/Ha+n6+AKAPYP+GjvGH/AEDdD/78Tf8Ax2j/AIaO8Yf9A3Q/+/E3/wAdrx+igD2D/ho7xh/0DdD/AO/E3/x2j/ho7xh/0DdD/wC/E3/x2vH6KAPYP+GjvGH/AEDdD/78Tf8Ax2j/AIaO8Yf9A3Q/+/E3/wAdrx+igD2D/ho7xh/0DdD/AO/E3/x2j/ho7xh/0DdD/wC/E3/x2vH6KAPYP+GjvGH/AEDdD/78Tf8Ax2j/AIaO8Yf9A3Q/+/E3/wAdrx+igD2D/ho7xh/0DdD/AO/E3/x2j/ho7xh/0DdD/wC/E3/x2vH6KAPYP+GjvGH/AEDdD/78Tf8Ax2j/AIaO8Yf9A3Q/+/E3/wAdrx+igD2D/ho7xh/0DdD/AO/E3/x2j/ho7xh/0DdD/wC/E3/x2vH6KAPYP+GjvGH/AEDdD/78Tf8Ax2j/AIaO8Yf9A3Q/+/E3/wAdrx+igD2D/ho7xh/0DdD/AO/E3/x2j/ho7xh/0DdD/wC/E3/x2vH6KAPYP+GjvGH/AEDdD/78Tf8Ax2j/AIaO8Yf9A3Q/+/E3/wAdrx+igD7/AKKKKACiiigAooooAKKKKACvgCvv+vgCgAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKAPv8AooooAKKKKACiiigAooooAK+AK+/6+AKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooA+/wCiiigAooooAKKKKACiiigAr4Ar7/r4AoAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigD7/AKKKKACiiigAooooAKKKKACvgCvv+vgCgAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKAPv8AooooAKKKKACiiigAooooAK+AK+/6+AKACiiigAooooAKKKKACiiigAr0jwt8FPEni7w5aa5YXulR2t1v2JPLIHG12Q5AjI6qe9eb19f/AAS/5JDoX/bx/wClElAHkH/DOPjD/oJaH/3/AJv/AI1R/wAM4+MP+glof/f+b/41X0/RQB8wf8M4+MP+glof/f8Am/8AjVH/AAzj4w/6CWh/9/5v/jVfT9FAHzB/wzj4w/6CWh/9/wCb/wCNUf8ADOPjD/oJaH/3/m/+NV9P0UAfMH/DOPjD/oJaH/3/AJv/AI1R/wAM4+MP+glof/f+b/41X0/RQB8wf8M4+MP+glof/f8Am/8AjVH/AAzj4w/6CWh/9/5v/jVfT9FAHzB/wzj4w/6CWh/9/wCb/wCNUf8ADOPjD/oJaH/3/m/+NV9P0UAFFFFABRRRQAUUUUAFFFFABXwBX3/XwBQAUUUUAFFFFABRRRQAUUUUAFfX/wAEv+SQ6F/28f8ApRJXyBX1/wDBL/kkOhf9vH/pRJQB6BRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAV8AV9/wBfAFABRRRQAUUUUAFFFFABRRRQAV9f/BL/AJJDoX/bx/6USV8gV9f/AAS/5JDoX/bx/wClElAHoFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABXwBX3/XwBQAUUUUAFFFFABRRRQAUUUUAFfX/wS/5JDoX/AG8f+lElfIFfX/wS/wCSQ6F/28f+lElAHoFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABXwBX3/XwBQAUUUUAFFFFABRRRQAUUUUAFfX/AMEv+SQ6F/28f+lElfIFfX/wS/5JDoX/AG8f+lElAHoFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABXwBRRQAUUUUAFFFFABRRRQAUUUUAFfX/wAEv+SQ6F/28f8ApRJRRQB6BRRRQAUUUUAFFFFABRRRQAUUUUAFFFFAH//ZAP/iArBJQ0NfUFJPRklMRQABAQAAAqBsY21zBDAAAG1udHJSR0IgWFlaIAflAAMAGQAPABwAMmFjc3BBUFBMAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAD21gABAAAAANMtbGNtcwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADWRlc2MAAAEgAAAAQGNwcnQAAAFgAAAANnd0cHQAAAGYAAAAFGNoYWQAAAGsAAAALHJYWVoAAAHYAAAAFGJYWVoAAAHsAAAAFGdYWVoAAAIAAAAAFHJUUkMAAAIUAAAAIGdUUkMAAAIUAAAAIGJUUkMAAAIUAAAAIGNocm0AAAI0AAAAJGRtbmQAAAJYAAAAJGRtZGQAAAJ8AAAAJG1sdWMAAAAAAAAAAQAAAAxlblVTAAAAJAAAABwARwBJAE0AUAAgAGIAdQBpAGwAdAAtAGkAbgAgAHMAUgBHAEJtbHVjAAAAAAAAAAEAAAAMZW5VUwAAABoAAAAcAFAAdQBiAGwAaQBjACAARABvAG0AYQBpAG4AAFhZWiAAAAAAAAD21gABAAAAANMtc2YzMgAAAAAAAQxCAAAF3v//8yUAAAeTAAD9kP//+6H///2iAAAD3AAAwG5YWVogAAAAAAAAb6AAADj1AAADkFhZWiAAAAAAAAAknwAAD4QAALbEWFlaIAAAAAAAAGKXAAC3hwAAGNlwYXJhAAAAAAADAAAAAmZmAADypwAADVkAABPQAAAKW2Nocm0AAAAAAAMAAAAAo9cAAFR8AABMzQAAmZoAACZnAAAPXG1sdWMAAAAAAAAAAQAAAAxlblVTAAAACAAAABwARwBJAE0AUG1sdWMAAAAAAAAAAQAAAAxlblVTAAAACAAAABwAcwBSAEcAQv/bAEMAAwICAwICAwMDAwQDAwQFCAUFBAQFCgcHBggMCgwMCwoLCw0OEhANDhEOCwsQFhARExQVFRUMDxcYFhQYEhQVFP/bAEMBAwQEBQQFCQUFCRQNCw0UFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFP/CABEIABUAGQMBEQACEQEDEQH/xAAXAAADAQAAAAAAAAAAAAAAAAAABwgJ/8QAFAEBAAAAAAAAAAAAAAAAAAAAAP/aAAwDAQACEAMQAAABpkSAANUzlHgBVRABpSB//8QAGxAAAQUBAQAAAAAAAAAAAAAABQAEBhc2AhD/2gAIAQEAAQUCIPeBrC6giuoIrqCKWZYey7JP6VNqlTalmWiep8//xAAUEQEAAAAAAAAAAAAAAAAAAAAw/9oACAEDAQE/AU//xAAUEQEAAAAAAAAAAAAAAAAAAAAw/9oACAECAQE/AU//xAAhEAAABQQDAQEAAAAAAAAAAAAAAQIDBQQ0k9IRdLIxEP/aAAgBAQAGPwKpq3CUbbDanVEn7wRci1kMaNxayGNG4tZDGjcTPTe8GKakbNJOPuJaSavnJnwLqPyL0F1H5F6CZ6b3gxDdxn2X7//EABcQAQEBAQAAAAAAAAAAAAAAAAERICH/2gAIAQEAAT8hLs8otSCp2GcWLErbs8qEQWDyu8WJWr//2gAMAwEAAgADAAAAEIABAAJP/8QAFBEBAAAAAAAAAAAAAAAAAAAAMP/aAAgBAwEBPxBP/8QAFBEBAAAAAAAAAAAAAAAAAAAAMP/aAAgBAgEBPxBP/8QAFxABAQEBAAAAAAAAAAAAAAAAAREgMf/aAAgBAQABPxAT69h7QKUBQsqdz06dNin17D2oAKoLLB5vp02bP//Z']}
    res = requests.post(self.url + '/vectorize', json=req_body)
    resBody = res.json()

    self.assertEqual(200, res.status_code)
    self.assertTrue(len(resBody['textVectors']) == 3)
    self.assertTrue(len(resBody['imageVectors']) == 2)
    self.assertTrue(resBody['imageVectors'][0] == resBody['imageVectors'][1])

    req_body = {'texts':['This is plane','Boeing 737','Blue plane']}
    res = requests.post(self.url + '/vectorize', json=req_body)
    resBody = res.json()

    self.assertEqual(200, res.status_code)
    self.assertTrue(len(resBody['textVectors']) == 3)
    self.assertTrue(len(resBody['imageVectors']) == 0)

    req_body = {'images':['/9j/4AAQSkZJRgABAQEASABIAAD/4QpKRXhpZgAASUkqAAgAAAAGABoBBQABAAAAVgAAABsBBQABAAAAXgAAACgBAwABAAAAAgAAADEBAgANAAAAZgAAADIBAgAUAAAAdAAAAGmHBAABAAAAiAAAAJoAAABIAAAAAQAAAEgAAAABAAAAR0lNUCAyLjEwLjE0AAAyMDIxOjAzOjI1IDE2OjI5OjQ3AAEAAaADAAEAAAABAAAAAAAAAAgAAAEEAAEAAAAAAQAAAQEEAAEAAADXAAAAAgEDAAMAAAAAAQAAAwEDAAEAAAAGAAAABgEDAAEAAAAGAAAAFQEDAAEAAAADAAAAAQIEAAEAAAAGAQAAAgIEAAEAAAA7CQAAAAAAAAgACAAIAP/Y/+AAEEpGSUYAAQEAAAEAAQAA/9sAQwAIBgYHBgUIBwcHCQkICgwUDQwLCwwZEhMPFB0aHx4dGhwcICQuJyAiLCMcHCg3KSwwMTQ0NB8nOT04MjwuMzQy/9sAQwEJCQkMCwwYDQ0YMiEcITIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIy/8AAEQgA1wEAAwEiAAIRAQMRAf/EAB8AAAEFAQEBAQEBAAAAAAAAAAABAgMEBQYHCAkKC//EALUQAAIBAwMCBAMFBQQEAAABfQECAwAEEQUSITFBBhNRYQcicRQygZGhCCNCscEVUtHwJDNicoIJChYXGBkaJSYnKCkqNDU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6g4SFhoeIiYqSk5SVlpeYmZqio6Slpqeoqaqys7S1tre4ubrCw8TFxsfIycrS09TV1tfY2drh4uPk5ebn6Onq8fLz9PX29/j5+v/EAB8BAAMBAQEBAQEBAQEAAAAAAAABAgMEBQYHCAkKC//EALURAAIBAgQEAwQHBQQEAAECdwABAgMRBAUhMQYSQVEHYXETIjKBCBRCkaGxwQkjM1LwFWJy0QoWJDThJfEXGBkaJicoKSo1Njc4OTpDREVGR0hJSlNUVVZXWFlaY2RlZmdoaWpzdHV2d3h5eoKDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uLj5OXm5+jp6vLz9PX29/j5+v/aAAwDAQACEQMRAD8A9/ooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAK+YP+GjvGH/QN0P8A78Tf/Ha+n6+AKAPYP+GjvGH/AEDdD/78Tf8Ax2j/AIaO8Yf9A3Q/+/E3/wAdrx+igD2D/ho7xh/0DdD/AO/E3/x2j/ho7xh/0DdD/wC/E3/x2vH6KAPYP+GjvGH/AEDdD/78Tf8Ax2j/AIaO8Yf9A3Q/+/E3/wAdrx+igD2D/ho7xh/0DdD/AO/E3/x2j/ho7xh/0DdD/wC/E3/x2vH6KAPYP+GjvGH/AEDdD/78Tf8Ax2j/AIaO8Yf9A3Q/+/E3/wAdrx+igD2D/ho7xh/0DdD/AO/E3/x2j/ho7xh/0DdD/wC/E3/x2vH6KAPYP+GjvGH/AEDdD/78Tf8Ax2j/AIaO8Yf9A3Q/+/E3/wAdrx+igD2D/ho7xh/0DdD/AO/E3/x2j/ho7xh/0DdD/wC/E3/x2vH6KAPYP+GjvGH/AEDdD/78Tf8Ax2j/AIaO8Yf9A3Q/+/E3/wAdrx+igD2D/ho7xh/0DdD/AO/E3/x2j/ho7xh/0DdD/wC/E3/x2vH6KAPYP+GjvGH/AEDdD/78Tf8Ax2j/AIaO8Yf9A3Q/+/E3/wAdrx+igD7/AKKKKACiiigAooooAKKKKACvgCvv+vgCgAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKAPv8AooooAKKKKACiiigAooooAK+AK+/6+AKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooA+/wCiiigAooooAKKKKACiiigAr4Ar7/r4AoAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigD7/AKKKKACiiigAooooAKKKKACvgCvv+vgC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']}
    res = requests.post(self.url + '/vectorize', json=req_body)
    resBody = res.json()

    self.assertEqual(200, res.status_code)
    self.assertTrue(len(resBody['textVectors']) == 0)
    self.assertTrue(len(resBody['imageVectors']) == 1)

  def testVectorizingPng(self):

    req_body = {'images':['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']}
    res = requests.post(self.url + '/vectorize', json=req_body)
    resBody = res.json()

    self.assertEqual(200, res.status_code)
    self.assertTrue(len(resBody['textVectors']) == 0)
    self.assertTrue(len(resBody['imageVectors']) == 1)

  def testVectorizingVideoModality(self):
    text_list=["A dog.", "A car", "A bird"]
    video_paths=["./test/VideoSamples/wind_noise.mp4"]
    req_body = {
      'texts': text_list,
      'video': convert_to_base64(video_paths)
    }
    res = requests.post(self.url + '/vectorize', json=req_body)
    resBody = res.json()

    self.assertEqual(200, res.status_code)
    self.assertTrue(len(resBody['textVectors']) == 3)
    self.assertTrue(len(resBody['videoVectors']) == 1)


def convert_to_base64(files: list) -> list:
  base64_encoded_files = []
  for file_path in files:
    base64_encoded_files.append(convert_file_to_base64(file_path))
  return base64_encoded_files


def convert_file_to_base64(file_path: str) -> str:
  with open(file_path, 'rb') as binary_file:
    binary_file_data = binary_file.read()
    base64_encoded_data = base64.b64encode(binary_file_data)
    base64_message = base64_encoded_data.decode('utf-8')
    return base64_message


def read_file_contents(file_paths: str) -> str:
  file_contents = []
  for file_path in file_paths:
    with open(file_path, 'r') as file:
      file_contents.append(file.read().rstrip())

  return file_contents


if __name__ == "__main__":
  unittest.main()
