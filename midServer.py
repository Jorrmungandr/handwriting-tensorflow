from http.server import HTTPServer, SimpleHTTPRequestHandler
from PIL import Image
import numpy as np
import PIL.ImageOps
import requests
import json
import base64

size = 28, 28

class HandRecogMidServer(SimpleHTTPRequestHandler):
    def res_send(self, res):
        response = bytes(res, 'utf8')

        self.send_response(200, 'ok')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Content-Length', str(len(response)))
        self.send_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'X-Requested-With')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

        self.wfile.write(response)

    def do_OPTIONS(self):
        self.res_send('CORS options')

    def do_POST(self):
        datalen = int(self.headers['Content-Length'])
        data = self.rfile.read(datalen)
        obj = json.loads(data)

        base64Data = obj['data']['base64']
        png = base64.b64decode(base64Data)
        with open('temp.png', 'wb') as f:
            f.write(png)
            # wrongImg = Image.open('temp.png')
            # rightImage = wrongImg.rotate(-90)
            # rightImage = rightImage.save('temp.png')
            f.close()

        img = Image.open('temp.png').resize(size, Image.ANTIALIAS).convert('L')
        img = PIL.ImageOps.invert(img)
        WIDTH, HEIGHT = img.size
        data = list(img.getdata())
        resultData = []
        for num in data:
            if (num > 110):
                resultData.append(255)
            else:
                resultData.append(0)
        twodData = [resultData[offset:offset+WIDTH] for offset in range(0, WIDTH*HEIGHT, WIDTH)]

        imgsData = []
        imgsData.append(twodData)

        for row in twodData:
          print(' '.join('{:3}'.format(value) for value in row))

        data = json.dumps({"signature_name": "serving_default", "instances": imgsData})
        headers = {"content-type": "application/json"}
        json_response = requests.post('http://localhost:8501/v1/models/handrecog:predict', data=data, headers=headers)
        predictions = json.loads(json_response.text)['predictions']
        highestChance = np.argmax(predictions[0])

        self.res_send('Isso provavelmente é um {}'.format(highestChance))


PORT = 8000
httpd = HTTPServer(('0.0.0.0', PORT), HandRecogMidServer)
print('Server running in port {}'.format(PORT))
httpd.serve_forever()
