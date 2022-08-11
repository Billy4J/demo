import asyncio

from sanic import Sanic
from sanic.response import text

app = Sanic(__name__)


@app.post('/post')
async def handler(request):
    print(request)
    return text('POST request ')


@app.get('/')
async def handler(request):
    print("得到请求")
    await asyncio.sleep(5)
    return text('hello')


if __name__ == '__main__':
    app.run("0.0.0.0", 5678)
