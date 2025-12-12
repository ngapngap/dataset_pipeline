# -*- coding: utf-8 -*-
"""
Run Dashboard Server

Usage:
    python run_dashboard.py
    python run_dashboard.py --port 8080
    python run_dashboard.py --host 0.0.0.0 --port 8080

Dashboard sẽ chạy tại http://localhost:8000 (mặc định)
"""

import argparse
import sys
import os

# Add current dir to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline Dashboard Server",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to bind (default: 127.0.0.1)"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind (default: 8000)"
    )
    
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development"
    )
    
    args = parser.parse_args()
    
    try:
        import uvicorn
    except ImportError:
        print("Error: uvicorn not installed")
        print("Install with: pip install uvicorn fastapi websockets jinja2")
        sys.exit(1)
    
    print(f"Starting Dashboard at http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop")
    print()
    
    uvicorn.run(
        "dashboard.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info"
    )


if __name__ == "__main__":
    main()

