class GlobalLogger:
    def __init__(self):
        self.dashboard = None
    
    def set_dashboard(self, dashboard):
        self.dashboard = dashboard
    
    def add_log(self, message):
        if self.dashboard:
            self.dashboard.add_log(message)

# Глобальный экземпляр
GLOBAL_LOGGER = GlobalLogger()
