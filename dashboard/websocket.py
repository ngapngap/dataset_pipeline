# -*- coding: utf-8 -*-
"""
WebSocket Manager - Quản lý connections và broadcast

Thread-safe connection management.
"""

from __future__ import annotations

import asyncio
from typing import List, Dict, Any
from fastapi import WebSocket


class ConnectionManager:
    """
    Quản lý WebSocket connections.
    
    Features:
    - Track active connections
    - Broadcast to all clients
    - Handle disconnect gracefully
    """
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self._lock = asyncio.Lock()
    
    async def connect(self, websocket: WebSocket) -> None:
        """Accept và lưu connection mới"""
        await websocket.accept()
        async with self._lock:
            self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket) -> None:
        """Remove connection"""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
    
    async def disconnect_all(self) -> None:
        """Disconnect tất cả connections"""
        async with self._lock:
            for connection in self.active_connections:
                try:
                    await connection.close()
                except Exception:
                    pass
            self.active_connections.clear()
    
    async def broadcast(self, data: Dict[str, Any]) -> None:
        """Broadcast data tới tất cả clients"""
        if not self.active_connections:
            return
        
        # Copy list để tránh modification during iteration
        connections = list(self.active_connections)
        
        for connection in connections:
            try:
                await connection.send_json(data)
            except Exception:
                # Connection failed, remove it
                self.disconnect(connection)
    
    async def send_to(self, websocket: WebSocket, data: Dict[str, Any]) -> bool:
        """Send data tới một client cụ thể"""
        try:
            await websocket.send_json(data)
            return True
        except Exception:
            self.disconnect(websocket)
            return False
    
    @property
    def connection_count(self) -> int:
        """Số connections đang active"""
        return len(self.active_connections)

