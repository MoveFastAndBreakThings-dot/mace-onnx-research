"""Run this to open the browser demo."""
import http.server, webbrowser, threading, os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

PORT = 8080
url = f"http://localhost:{PORT}/demo.html"

threading.Timer(1.0, lambda: webbrowser.open(url)).start()
print(f"Opening {url}")
print("Press Ctrl+C to stop.")

http.server.test(HandlerClass=http.server.SimpleHTTPRequestHandler, port=PORT)
