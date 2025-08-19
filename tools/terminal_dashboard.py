import asyncio
import requests
import json
from datetime import datetime
from prompt_toolkit import Application
from prompt_toolkit.layout import Layout, HSplit, VSplit
from prompt_toolkit.widgets import Frame, TextArea, Label
from prompt_toolkit.layout.containers import Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.key_binding import KeyBindings
import threading
import time

class TerminalDashboard:
    def __init__(self, prometheus_url="http://localhost:9090"):
        self.prometheus_url = prometheus_url
        self.running = True
        
        # UI Components
        self.title_label = Label("QIKI Neural Engine - Terminal Dashboard")
        self.status_label = Label("Status: Running")
        self.metrics_text = TextArea(text="Loading metrics...", height=10)
        self.logs_text = TextArea(text="Waiting for logs...", height=10)
        self.footer_label = Label("Press 'q' to quit | Auto-refresh: 2s")
        
        # Layout
        self.root_container = HSplit([
            Frame(self.title_label, style="class:title"),
            Frame(self.status_label, style="class:status"),
            Frame(Label("Metrics"), body=self.metrics_text),
            Frame(Label("Recent Logs"), body=self.logs_text),
            Frame(self.footer_label, style="class:footer")
        ])
        
        self.layout = Layout(self.root_container)
        
        # Key bindings
        self.bindings = KeyBindings()
        
        @self.bindings.add('q')
        def _(event):
            self.running = False
            event.app.exit()
            
        @self.bindings.add('c-c')
        def _(event):
            self.running = False
            event.app.exit()
        
        # App
        self.app = Application(
            layout=self.layout,
            key_bindings=self.bindings,
            full_screen=True
        )
        
        # Data storage
        self.logs = []
        
    def query_prometheus(self, query):
        """Query Prometheus API"""
        try:
            url = f"{self.prometheus_url}/api/v1/query"
            response = requests.get(url, params={'query': query}, timeout=2)
            if response.status_code == 200:
                data = response.json()
                if data['status'] == 'success' and data['data']['result']:
                    return float(data['data']['result'][0]['value'][1])
            return 0.0
        except:
            return 0.0
    
    def update_metrics(self):
        """Update metrics display"""
        try:
            # Get metrics
            inference_rate = self.query_prometheus('rate(ne_inference_total[1m])')
            latency = self.query_prometheus('histogram_quantile(0.95, rate(ne_inference_duration_seconds_bucket[1m]))')
            avg_confidence = self.query_prometheus('avg(ne_avg_confidence)')
            
            # Format metrics text
            metrics_text = f"""QIKI Neural Engine Metrics
====================================================================================================
Inference Rate:     {inference_rate:.2f} infers/sec
95th Latency:       {latency*1000:.2f} ms
Avg Confidence:     {avg_confidence:.3f}
Safety Blocks:      {self.query_prometheus('ne_safety_blocks_total')}
Degradations:       {self.query_prometheus('ne_degradation_to_rule')}
====================================================================================================
"""
            self.metrics_text.text = metrics_text
            
        except Exception as e:
            self.metrics_text.text = f"Error fetching metrics: {str(e)}"
    
    def add_log(self, log_entry):
        """Add log entry to display"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.logs.append(f"[{timestamp}] {log_entry}")
        # Keep only last 50 logs
        self.logs = self.logs[-50:]
        self.logs_text.text = "\n".join(self.logs)
    
    def auto_refresh(self):
        """Auto-refresh metrics"""
        while self.running:
            try:
                self.update_metrics()
                time.sleep(2)
            except:
                pass
    
    def run(self):
        """Run the dashboard"""
        # Start auto-refresh thread
        refresh_thread = threading.Thread(target=self.auto_refresh, daemon=True)
        refresh_thread.start()
        
        # Run application
        self.app.run()
        
        self.running = False

# Пример использования
if __name__ == "__main__":
    dashboard = TerminalDashboard()
    
    # Добавим тестовые логи
    def simulate_logs():
        import time
        import random
        logs = [
            "NeuralEngine: Proposal generated (HOLD_POSITION, conf=0.87)",
            "SafetyShield: Action validated OK",
            "ProposalEvaluator: Selected priority=0.92",
            "FSM: State changed to ACTIVE",
            "BIOS: All systems OK",
            "Timeout: 6.2ms (within budget)",
            "Warning: Low confidence (0.45) - degrading to RuleEngine"
        ]
        while dashboard.running:
            if random.random() < 0.3:  # 30% chance per 0.5s
                log = random.choice(logs)
                dashboard.add_log(log)
            time.sleep(0.5)
    
    # Start log simulation
    log_thread = threading.Thread(target=simulate_logs, daemon=True)
    log_thread.start()
    
    # Run dashboard
    dashboard.run()
