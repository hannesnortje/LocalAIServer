import sys
import signal
import threading
import time
import logging
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from werkzeug.serving import make_server

# Add package root to Python path for imports
package_root = Path(__file__).parent.parent
sys.path.insert(0, str(package_root))

from local_ai_server.server import app, get_ssl_context, HTTP_PORT, HTTPS_PORT
from local_ai_server import __version__

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ServerRunner:
    def __init__(self):
        self.shutdown_event = threading.Event()
        self.shutdown_in_progress = False
        self.servers = []

    def run_server(self, app, port, ssl_context=None):
        try:
            server = make_server(
                '0.0.0.0', 
                port, 
                app, 
                ssl_context=ssl_context,
                threaded=True
            )
            self.servers.append(server)
            server.serve_forever()
        except Exception as e:
            logger.error(f"Server error: {e}")
        finally:
            self.shutdown_event.set()

    def shutdown(self):
        """Graceful shutdown of all servers"""
        self.shutdown_in_progress = True
        for server in self.servers:
            server.shutdown()
        self.shutdown_event.set()

def main():
    """Start HTTP and HTTPS servers"""
    runner = ServerRunner()
    
    try:
        # Check for version flag
        if len(sys.argv) > 1 and sys.argv[1] == "--version":
            print(f"Local AI Server v{__version__}")
            return 0

        ssl_context, cert_file, key_file = get_ssl_context()
        
        # Register signal handlers
        for sig in [signal.SIGINT, signal.SIGTERM]:
            signal.signal(sig, lambda s, f: runner.shutdown())
        
        print("Starting servers:")
        print(f"HTTP  server at http://localhost:{HTTP_PORT}")
        print(f"HTTPS server at https://localhost:{HTTPS_PORT}")
        print("API documentation available at:")
        print(f"- http://localhost:{HTTP_PORT}/docs")
        print(f"- https://localhost:{HTTPS_PORT}/docs")
        print("\nPress Ctrl+C to stop")
        
        # Start both HTTP and HTTPS servers
        executor = ThreadPoolExecutor(max_workers=2)
        http_future = executor.submit(runner.run_server, app, HTTP_PORT)
        https_future = executor.submit(runner.run_server, app, HTTPS_PORT, ssl_context)
        
        # Wait for shutdown signal
        runner.shutdown_event.wait()
        
        print("Shutting down servers...")
        executor.shutdown(wait=True)
        print("Server shutdown completed")
        return 0
            
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    sys.exit(main())
