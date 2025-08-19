import json
import asyncio
import nats

class NATSLogger:
    def __init__(self, nats_url="nats://localhost:4222"):
        self.nats_url = nats_url
        self.nc = None

    async def connect(self):
        try:
            self.nc = await nats.connect(self.nats_url)
        except Exception as e:
            print(f"[NATS] Connection failed: {e}")

    async def log_proposal(self, proposal_data):
        if self.nc:
            try:
                await self.nc.publish("qiki.neural.proposals", json.dumps(proposal_data).encode())
            except Exception as e:
                print(f"[NATS] Publish failed: {e}")

    async def close(self):
        if self.nc:
            await self.nc.close()
