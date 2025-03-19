import uvicorn
import asyncio
import sys
import signal
import threading
import time
import logging
import os
import _thread
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import ssl

# Add package root to Python path for imports
package_root = Path(__file__).parent.parent
sys.path.insert(0, str(package_root))

from local_ai_server.server import app, get_ssl_context, HTTP_PORT, HTTPS_PORT
from local_ai_server import __version__

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global shutdown flag and process status
shutdown_event = threading.Event()
shutdown_in_progress = False

# Add kill timer to force exit after a delay
def force_exit_timer():
    """Force exit after 5 seconds if graceful shutdown fails"""
    time.sleep(5)
    print("Force exiting after timeout...")
    os._exit(1)

def run_server(config):
    """Run a uvicorn server in a separate thread"""
    try:
        server = uvicorn.Server(config)
        server.run()
    except Exception as e:
        logger.error(f"Server error: {e}")
    finally:
        # Signal main thread when server exits
        shutdown_event.set()

def handle_signal(sig, frame):
    """Handle shutdown signals"""
    global shutdown_in_progress
    
    if shutdown_in_progress:
        print("\nForce exiting...")
        os._exit(0)
        return
    
    shutdown_in_progress = True
    
    if sig == signal.SIGINT:
        print(f"\nReceived SIGINT. Shutting down...")
    elif sig == signal.SIGTERM:
        print(f"\nReceived SIGTERM. Shutting down...")
    
    # Set shutdown event to stop the main thread
    shutdown_event.set()
    
    # Start force exit timer - will exit if shutdown takes too long
    _thread.start_new_thread(force_exit_timer, ())

def main():
    """Start HTTP and HTTPS servers"""
    try:
        # Check for version flag
        if len(sys.argv) > 1 and sys.argv[1] == "--version":
            print(f"Local AI Server v{__version__}")
            return 0

        # Generate SSL certificates
        ssl_context, cert_file, key_file = get_ssl_context()
        
        # Configure HTTP and HTTPS servers
        http_config = uvicorn.Config(
            app=app,
            host="0.0.0.0",
            port=HTTP_PORT,
            log_level="info",
            timeout_keep_alive=5,
            timeout_graceful_shutdown=3,
            limit_concurrency=100
        )
        
        https_config = uvicorn.Config(
            app=app,
            host="0.0.0.0",
            port=HTTPS_PORT,
            ssl_keyfile=key_file,
            ssl_certfile=cert_file,
            log_level="info",
            timeout_keep_alive=5,
            timeout_graceful_shutdown=3,
            limit_concurrency=100
        )
        
        # Register signal handlers
        signal.signal(signal.SIGINT, handle_signal)
        signal.signal(signal.SIGTERM, handle_signal)
        
        print("Starting servers:")
        print(f"HTTP  server at http://localhost:{HTTP_PORT}")
        print(f"HTTPS server at https://localhost:{HTTPS_PORT}")
        print("API documentation available at:")
        print(f"- http://localhost:{HTTP_PORT}/docs")
        print(f"- https://localhost:{HTTPS_PORT}/docs")
        print("\nPress Ctrl+C to stop")
        
        # Start both HTTP and HTTPS servers
        executor = ThreadPoolExecutor(max_workers=2)
        http_future = executor.submit(run_server, http_config)
        https_future = executor.submit(run_server, https_config)
        
        # Wait for shutdown signal
        try:
            # Wait for shutdown event or server failure
            while not shutdown_event.is_set():
                time.sleep(0.1)
                
                # Check if either server failed
                for server_name, future in [("HTTP", http_future), ("HTTPS", https_future)]:
                    if future.done() and future.exception():
                        logger.error(f"{server_name} server error: {future.exception()}")
                        # Continue even if one server fails
            
            print("Shutting down server...")
            
            # Try a graceful shutdown by stopping executor
            executor.shutdown(wait=False, cancel_futures=True)
            
            # Wait briefly for shutdown
            time.sleep(1)
            
            # Always exit cleanly - force_exit_timer will kill if needed
            print("Server shutdown initiated")
            return 0
            
        except KeyboardInterrupt:
            # Second Ctrl+C - force exit
            print("Force exiting...")
            os._exit(0)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    # Exit handler to ensure we always exit cleanly
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("Interrupted, exiting...")
        os._exit(0)
