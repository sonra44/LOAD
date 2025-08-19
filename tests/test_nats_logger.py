import unittest
from unittest.mock import AsyncMock, patch
import asyncio
from core.nats_logger import NATSLogger

class TestNATSLogger(unittest.TestCase):
    def setUp(self):
        self.logger = NATSLogger()

    @patch('core.nats_logger.nats.connect')
    def test_connect(self, mock_connect):
        mock_connect.return_value = AsyncMock()
        async def run_test():
            await self.logger.connect()
            mock_connect.assert_called_once()
        
        asyncio.run(run_test())

    @patch('core.nats_logger.nats.connect')
    def test_log_proposal(self, mock_connect):
        mock_nc = AsyncMock()
        mock_connect.return_value = mock_nc
        
        async def run_test():
            await self.logger.connect()
            await self.logger.log_proposal({"test": "data"})
            mock_nc.publish.assert_called_once()
        
        asyncio.run(run_test())

if __name__ == "__main__":
    unittest.main()
