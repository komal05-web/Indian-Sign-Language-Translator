import json
from channels.generic.websocket import AsyncWebsocketConsumer
from asgiref.sync import sync_to_async
from .predictor_engine import ISLEngine


class PredictorConsumer(AsyncWebsocketConsumer):

    async def connect(self):
        self.engine = ISLEngine(mode="WORD")
        self.active = True
        await self.accept()
        await self.send(json.dumps({"type": "connected", "mode": "WORD"}))

    async def disconnect(self, code):
        self.active = False
        if hasattr(self, "engine"):
            await sync_to_async(self.engine.close)()

    async def receive(self, text_data=None, bytes_data=None):
        if not self.active:
            return

        if bytes_data:
            result = await sync_to_async(self.engine.process_frame)(bytes_data)
            result["type"] = "prediction"
            await self.send(json.dumps(result))
            return

        if text_data:
            msg = json.loads(text_data)
            cmd = msg.get("cmd", "")

            if cmd == "SET_MODE":
                mode = msg.get("mode", "WORD")
                self.engine.set_mode(mode)
                await self.send(json.dumps({"type": "mode_changed", "mode": mode}))

            elif cmd == "TOGGLE_RECORDING":
                result = self.engine.toggle_recording()
                result["type"] = "recording_status"
                await self.send(json.dumps(result))

            elif cmd in ("SPACE", "BACKSPACE", "CLEAR", "ENTER"):
                result = self.engine.word_command(cmd)
                result["type"] = "word_update"
                if cmd == "ENTER":
                    result["speak"] = result.get("matched") or result.get("sentence", "")
                await self.send(json.dumps(result))

            elif cmd == "PING":
                await self.send(json.dumps({"type": "pong"}))
