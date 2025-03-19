import uvicorn
import asyncio
import sys
import signal
from pathlib import Path
import logging

# Add package root to Python path for imports
package_root = Path(__file__).parent.parent
sys.path.insert(0, str(package_root))

from local_ai_server.server import app, get_ssl_context, HTTP_PORT, HTTPS_PORT

class ServerManager:
    def __init__(self):
        self.servers = []
        self.should_exit = False
        self.shutdown_event = asyncio.Event()
        self._shutdown_complete = False

    async def shutdown(self, sig=None):
        """Cleanup tasks tied to the service's shutdown."""
        if self._shutdown_complete:
            return
            
        if sig:
            print(f"\nReceived exit signal {sig.name}...")
        
        self.should_exit = True
        self.shutdown_event.set()

        # First, stop accepting new connections
        for server in self.servers:
            server.should_exit = True
        
        # Wait a bit for current requests to finish
        await asyncio.sleep(0.1)
        
        # Shut down servers
        shutdown_tasks = [server.shutdown() for server in self.servers]
        await asyncio.gather(*shutdown_tasks, return_exceptions=True)
        
        # Cancel remaining tasks
        pending = asyncio.all_tasks()
        for task in pending:
            if task is not asyncio.current_task():
                task.cancel()
        
        self._shutdown_complete = True

    async def run_servers(self):
        ssl_context, cert_file, key_file = get_ssl_context()
        
        config_http = uvicorn.Config(
            app, 
            host="0.0.0.0", 
            port=HTTP_PORT,
            log_level=logging.INFO
        )
        config_https = uvicorn.Config(
            app, 
            host="0.0.0.0", 
            port=HTTPS_PORT,
            ssl_keyfile=key_file,
            ssl_certfile=cert_file,
            log_level=logging.INFO
        )
        
        server_http = uvicorn.Server(config_http)
        server_https = uvicorn.Server(config_https)
        self.servers = [server_http, server_https]
        
        print("Starting servers:")
        print(f"HTTP  server at http://localhost:{HTTP_PORT}")
        print(f"HTTPS server at https://localhost:{HTTPS_PORT}")
        print("API documentation available at:")
        print(f"- http://localhost:{HTTP_PORT}/docs")
        print(f"- https://localhost:{HTTPS_PORT}/docs")
        print("\nPress Ctrl+C to stop")
        
        try:
            servers_task = asyncio.gather(
                server_http.serve(),
                server_https.serve()
            )
            await servers_task
        except asyncio.CancelledError:
            await self.shutdown()
            servers_task.cancel()
            try:
                await servers_task
            except asyncio.CancelledError:
                pass

def main():
    server_manager = ServerManager()
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    def handle_signal(sig):
        loop.create_task(server_manager.shutdown(sig))
        # Give it a moment to shut down gracefully
        loop.call_later(0.5, loop.stop)
    
    # Handle signals
    signals = (signal.SIGHUP, signal.SIGTERM, signal.SIGINT)
    for s in signals:
        loop.add_signal_handler(
            s, lambda s=s: handle_signal(s)
        )
    
    try:
        loop.run_until_complete(server_manager.run_servers())
        return 0
    except KeyboardInterrupt:
        loop.run_until_complete(server_manager.shutdown())
        return 0
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()

if __name__ == "__main__":
    sys.exit(main())
