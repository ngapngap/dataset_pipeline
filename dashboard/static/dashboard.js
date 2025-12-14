/**
 * Pipeline Dashboard - Real-time Updates
 */

class Dashboard {
    constructor() {
        this.ws = null;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 10;
        this.reconnectDelay = 1000;
        
        this.elements = {
            connectionStatus: document.getElementById('connection-status'),
            currentStep: document.getElementById('current-step'),
            elapsedTime: document.getElementById('elapsed-time'),
            progressFill: document.getElementById('progress-fill'),
            chunksProcessed: document.getElementById('chunks-processed'),
            chunksTotal: document.getElementById('chunks-total'),
            progressPercent: document.getElementById('progress-percent'),
            processingRate: document.getElementById('processing-rate'),
            estimatedRemaining: document.getElementById('estimated-remaining'),
            docsExtracted: document.getElementById('docs-extracted'),
            docsExtractedRow: document.getElementById('docs-extracted-row'),
            qaGenerated: document.getElementById('qa-generated'),
            qaGood: document.getElementById('qa-good'),
            qaBad: document.getElementById('qa-bad'),
            qaRescued: document.getElementById('qa-rescued'),
            qaGoodRate: document.getElementById('qa-good-rate'),
            cacheHits: document.getElementById('cache-hits'),
            cacheMisses: document.getElementById('cache-misses'),
            cacheHitRate: document.getElementById('cache-hit-rate'),
            chunksSuccess: document.getElementById('chunks-success'),
            chunksFailed: document.getElementById('chunks-failed'),
            errorCount: document.getElementById('error-count'),
            errorList: document.getElementById('error-list'),
            lastUpdate: document.getElementById('last-update')
        };
        
        this.connect();
    }
    
    connect() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws`;
        
        console.log('Connecting to WebSocket:', wsUrl);
        
        this.ws = new WebSocket(wsUrl);
        
        this.ws.onopen = () => {
            console.log('WebSocket connected');
            this.reconnectAttempts = 0;
            this.setConnectionStatus('connected');
        };
        
        this.ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                this.updateDashboard(data);
            } catch (e) {
                console.error('Error parsing message:', e);
            }
        };
        
        this.ws.onclose = () => {
            console.log('WebSocket disconnected');
            this.setConnectionStatus('disconnected');
            this.scheduleReconnect();
        };
        
        this.ws.onerror = (error) => {
            console.error('WebSocket error:', error);
            this.setConnectionStatus('disconnected');
        };
    }
    
    scheduleReconnect() {
        if (this.reconnectAttempts >= this.maxReconnectAttempts) {
            console.log('Max reconnect attempts reached');
            return;
        }
        
        this.reconnectAttempts++;
        const delay = this.reconnectDelay * Math.pow(1.5, this.reconnectAttempts - 1);
        
        console.log(`Reconnecting in ${delay}ms (attempt ${this.reconnectAttempts})`);
        
        setTimeout(() => {
            if (this.ws.readyState === WebSocket.CLOSED) {
                this.connect();
            }
        }, delay);
    }
    
    setConnectionStatus(status) {
        const el = this.elements.connectionStatus;
        el.className = `status ${status}`;
        
        const textEl = el.querySelector('.status-text');
        switch (status) {
            case 'connected':
                textEl.textContent = 'Đang kết nối';
                break;
            case 'disconnected':
                textEl.textContent = 'Mất kết nối';
                break;
            default:
                textEl.textContent = 'Đang kết nối...';
        }
    }
    
    updateDashboard(data) {
        // Current step
        this.elements.currentStep.textContent = this.formatStepName(data.current_step);
        this.elements.elapsedTime.textContent = data.elapsed_formatted || '--:--';

        // Progress
        if (data.chunks) {
            const chunks = data.chunks;
            this.elements.chunksProcessed.textContent = chunks.processed;
            this.elements.chunksTotal.textContent = chunks.total;
            this.elements.progressPercent.textContent = chunks.progress_percent.toFixed(1);
            this.elements.progressFill.style.width = `${chunks.progress_percent}%`;
            this.elements.chunksSuccess.textContent = chunks.success || chunks.processed;
            this.elements.chunksFailed.textContent = chunks.failed;
        }

        // Docs extracted
        if (data.docs_extracted !== undefined && data.docs_extracted > 0) {
            this.elements.docsExtractedRow.style.display = 'block';
            this.elements.docsExtracted.textContent = data.docs_extracted;
        } else {
            this.elements.docsExtractedRow.style.display = 'none';
        }

        // Performance
        if (data.performance) {
            this.elements.processingRate.textContent = data.performance.processing_rate.toFixed(1);
            this.elements.estimatedRemaining.textContent = data.performance.estimated_remaining;
        }

        // QA Stats
        if (data.qa) {
            this.elements.qaGenerated.textContent = data.qa.generated;
            this.elements.qaGood.textContent = data.qa.good;
            this.elements.qaBad.textContent = data.qa.bad;
            this.elements.qaRescued.textContent = data.qa.rescued || 0;
            this.elements.qaGoodRate.textContent = data.qa.good_rate.toFixed(1) + '%';
        }

        // Cache Stats
        if (data.cache) {
            this.elements.cacheHits.textContent = data.cache.hits;
            this.elements.cacheMisses.textContent = data.cache.misses;
            this.elements.cacheHitRate.textContent = data.cache.hit_rate.toFixed(1) + '%';
        }

        // Errors
        if (data.errors) {
            this.elements.errorCount.textContent = data.errors.count;
            this.updateErrorList(data.errors.recent);
        }

        // Last update
        this.elements.lastUpdate.textContent = new Date().toLocaleTimeString();
    }
    
    formatStepName(step) {
        const stepNames = {
            'idle': 'Cho khoi dong',
            'extract': 'Dang trich xuat...',
            'extract_done': 'Trich xuat xong',
            'generate': 'Dang sinh Q&A...',
            'generate_done': 'Sinh Q&A xong',
            'evaluate': 'Dang danh gia...',
            'evaluate_done': 'Danh gia xong',
            'split': 'Dang chia dataset...',
            'split_done': 'Chia dataset xong',
            'export': 'Dang xuat file...',
            'completed': 'HOAN THANH!'
        };
        return stepNames[step] || step;
    }
    
    updateErrorList(errors) {
        if (!errors || errors.length === 0) {
            this.elements.errorList.innerHTML = '<div class="no-errors">Không có lỗi</div>';
            return;
        }
        
        const html = errors.reverse().map(err => `
            <div class="error-item">
                <span class="error-time">${this.formatTime(err.timestamp)}</span>
                <span class="error-type">${err.type}</span>
                <span class="error-message">${this.escapeHtml(err.message)}</span>
            </div>
        `).join('');
        
        this.elements.errorList.innerHTML = html;
    }
    
    formatTime(timestamp) {
        try {
            return new Date(timestamp).toLocaleTimeString();
        } catch {
            return timestamp;
        }
    }
    
    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
}

// Initialize dashboard when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.dashboard = new Dashboard();
});

