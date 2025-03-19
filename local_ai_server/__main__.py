import uvicorn
import asyncio
import sys
from pathlib import Path

# Add package root to Python path for imports
package_root = Path(__file__).parent.parent
sys.path.insert(0, str(package_root))

from local_ai_server.server import app, get_ssl_context, HTTP_PORT, HTTPS_PORT

async def run_servers():
    ssl_context, cert_file, key_file = get_ssl_context()
    
    config_http = uvicorn.Config(app, host="0.0.0.0", port=HTTP_PORT)
    config_https = uvicorn.Config(
        app, 
        host="0.0.0.0", 
        port=HTTPS_PORT,
        ssl_keyfile=key_file,
        ssl_certfile=cert_file
    )
    
    server_http = uvicorn.Server(config_http)
    server_https = uvicorn.Server(config_https)
    
    print("Starting servers:")
    print(f"HTTP  server at http://localhost:{HTTP_PORT}")
    print(f"HTTPS server at https://localhost:{HTTPS_PORT}")
    print("API documentation available at:")
    print(f"- http://localhost:{HTTP_PORT}/docs")
    print(f"- https://localhost:{HTTPS_PORT}/docs")
    
    await asyncio.gather(
        server_http.serve(),
        server_https.serve()
    )

def main():
    asyncio.run(run_servers())

if __name__ == "__main__":
    main()
