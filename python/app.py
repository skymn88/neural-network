"""Start the local UI; no hosted service and no external data transfer."""

import argparse
import threading
import webbrowser
import uvicorn

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    if not args.no_browser:
        timer = threading.Timer(
            1.5, lambda: webbrowser.open(f"http://127.0.0.1:{args.port}")
        )
        timer.daemon = True
        timer.start()
    uvicorn.run("actuator.server:app", host="127.0.0.1", port=args.port)
