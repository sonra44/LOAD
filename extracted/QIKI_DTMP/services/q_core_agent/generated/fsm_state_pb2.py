"""Заглушка protobuf сообщений FSM для тестов."""
from dataclasses import dataclass, field
from enum import IntEnum
from typing import List, Dict
import json

from google.protobuf.timestamp_pb2 import Timestamp

from .common_types_pb2 import UUID


class FSMStateEnum(IntEnum):
    FSM_STATE_UNSPECIFIED = 0
    BOOTING = 1
    IDLE = 2
    ACTIVE = 3
    ERROR_STATE = 4
    SHUTDOWN = 5


class FSMTransitionStatus(IntEnum):
    FSM_TRANSITION_STATUS_UNSPECIFIED = 0
    SUCCESS = 1
    FAILED = 2
    PENDING = 3


@dataclass
class StateTransition:
    from_state: int = FSMStateEnum.FSM_STATE_UNSPECIFIED
    to_state: int = FSMStateEnum.FSM_STATE_UNSPECIFIED
    trigger_event: str = ""
    status: int = FSMTransitionStatus.FSM_TRANSITION_STATUS_UNSPECIFIED
    error_message: str = ""
    timestamp: Timestamp = field(default_factory=Timestamp)


@dataclass
class FsmStateSnapshot:
    snapshot_id: UUID = field(default_factory=UUID)
    fsm_instance_id: UUID = field(default_factory=UUID)
    timestamp: Timestamp = field(default_factory=Timestamp)
    current_state: int = FSMStateEnum.FSM_STATE_UNSPECIFIED
    source_module: str = ""
    attempt_count: int = 0
    history: List[StateTransition] = field(default_factory=list)
    context_data: Dict[str, str] = field(default_factory=dict)
    state_metadata: Dict[str, str] = field(default_factory=dict)

    def SerializeToString(self) -> bytes:
        """Простая сериализация в JSON."""
        def transition_dict(t: StateTransition) -> Dict[str, object]:
            return {
                'from_state': int(t.from_state),
                'to_state': int(t.to_state),
                'trigger_event': t.trigger_event,
                'status': int(t.status),
                'error_message': t.error_message,
                'timestamp': {'seconds': t.timestamp.seconds, 'nanos': t.timestamp.nanos},
            }

        data = {
            'snapshot_id': self.snapshot_id.value,
            'fsm_instance_id': self.fsm_instance_id.value,
            'timestamp': {'seconds': self.timestamp.seconds, 'nanos': self.timestamp.nanos},
            'current_state': int(self.current_state),
            'source_module': self.source_module,
            'attempt_count': self.attempt_count,
            'history': [transition_dict(t) for t in self.history],
            'context_data': self.context_data,
            'state_metadata': self.state_metadata,
        }
        return json.dumps(data).encode()

    def ParseFromString(self, data: bytes) -> None:
        """Простая десериализация из JSON."""
        obj = json.loads(data.decode())
        self.snapshot_id.value = obj.get('snapshot_id', '')
        self.fsm_instance_id.value = obj.get('fsm_instance_id', '')
        ts = obj.get('timestamp', {})
        self.timestamp.seconds = ts.get('seconds', 0)
        self.timestamp.nanos = ts.get('nanos', 0)
        self.current_state = obj.get('current_state', 0)
        self.source_module = obj.get('source_module', '')
        self.attempt_count = obj.get('attempt_count', 0)
        self.history.clear()
        for h in obj.get('history', []):
            tr = StateTransition(
                from_state=h.get('from_state', 0),
                to_state=h.get('to_state', 0),
                trigger_event=h.get('trigger_event', ''),
                status=h.get('status', 0),
                error_message=h.get('error_message', ''),
            )
            ts = h.get('timestamp', {})
            tr.timestamp.seconds = ts.get('seconds', 0)
            tr.timestamp.nanos = ts.get('nanos', 0)
            self.history.append(tr)
        self.context_data = {str(k): str(v) for k, v in obj.get('context_data', {}).items()}
        self.state_metadata = {str(k): str(v) for k, v in obj.get('state_metadata', {}).items()}


def MessageToDict(proto: FsmStateSnapshot) -> Dict[str, object]:
    """Утилита для преобразования снапшота в dict."""
    return json.loads(proto.SerializeToString())

