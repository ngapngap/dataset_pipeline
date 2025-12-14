# -*- coding: utf-8 -*-
"""
FastAPI Application - Dashboard server

Endpoints:
- GET / : Dashboard UI
- GET /api/stats : Current stats
- WS /ws : WebSocket for real-time updates
"""

from __future__ import annotations

import os
import asyncio
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

from .file_metrics import FileMetricsReader, get_file_metrics
from .websocket import ConnectionManager


def create_app(output_dir: str = "./output") -> FastAPI:
    """
    Create FastAPI application
    
    Args:
        output_dir: Output directory của pipeline để đọc metrics
        
    Returns:
        FastAPI app instance
    """
    app = FastAPI(
        title="Pipeline Dashboard",
        description="Real-time monitoring for Dataset Pipeline",
        version="1.0.0"
    )
    
    # Dùng FileMetricsReader để đọc từ files (hỗ trợ multi-process)
    app.state.metrics = get_file_metrics(output_dir)
    
    # WebSocket manager
    app.state.ws_manager = ConnectionManager()
    
    # Static files và templates
    dashboard_dir = Path(__file__).parent
    templates_dir = dashboard_dir / "templates"
    static_dir = dashboard_dir / "static"
    
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
    
    templates = Jinja2Templates(directory=str(templates_dir))
    
    # =========================================================================
    # ROUTES
    # =========================================================================
    
    @app.get("/", response_class=HTMLResponse)
    async def dashboard(request: Request):
        """Dashboard home page"""
        return templates.TemplateResponse(
            "index.html",
            {"request": request}
        )
    
    @app.get("/api/stats")
    async def get_stats():
        """Get current pipeline stats"""
        return app.state.metrics.get_stats()
    
    @app.post("/api/reset")
    async def reset_stats():
        """Reset all stats"""
        app.state.metrics.reset()
        return {"status": "ok", "message": "Stats reset"}
    
    # =========================================================================
    # WEBSOCKET
    # =========================================================================
    
    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        """WebSocket endpoint for real-time updates"""
        await app.state.ws_manager.connect(websocket)
        
        try:
            # Send initial stats
            stats = app.state.metrics.get_stats()
            await websocket.send_json(stats)
            
            # Keep connection alive and send updates
            while True:
                # Wait for either a message or timeout
                try:
                    # Wait for ping from client
                    data = await asyncio.wait_for(
                        websocket.receive_text(),
                        timeout=1.0
                    )
                    
                    if data == "ping":
                        await websocket.send_text("pong")
                    elif data == "stats":
                        stats = app.state.metrics.get_stats()
                        await websocket.send_json(stats)
                        
                except asyncio.TimeoutError:
                    # No message, send periodic update
                    stats = app.state.metrics.get_stats()
                    await websocket.send_json(stats)
                    
        except WebSocketDisconnect:
            app.state.ws_manager.disconnect(websocket)
        except Exception:
            app.state.ws_manager.disconnect(websocket)
    
    # =========================================================================
    # STARTUP / SHUTDOWN
    # =========================================================================
    
    @app.on_event("startup")
    async def startup_event():
        """On application startup"""
        # File-based metrics không cần callback
        # WebSocket sẽ poll files mỗi giây
        pass
    
    @app.on_event("shutdown")
    async def shutdown_event():
        """On application shutdown"""
        # Close all WebSocket connections
        await app.state.ws_manager.disconnect_all()
    
    return app


# Default app for uvicorn
app = create_app()

