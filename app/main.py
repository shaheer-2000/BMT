from app.app import app
import nest_asyncio
from pyngrok import ngrok
import uvicorn

ngrok_tunnel = ngrok.connect(8000)
print(f"Public URL: {ngrok_tunnel.public_url}")

nest_asyncio.apply()
uvicorn.run(app, port=8000)