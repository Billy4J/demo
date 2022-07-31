from sanic import Sanic
from sanic.response import text
import cv2
import numpy

app = Sanic(__name__)


@app.post('/post')
async def handler(request):
    print(request)
    return text('POST request ')

@app.get('/')
async def handler(request):
    print(request)
    return text('POST request ')

if __name__ == '__main__':
    app.run("0.0.0.0",5678)