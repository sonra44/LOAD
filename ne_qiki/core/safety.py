from shared.models import Proposal, ActuatorCommand
from collections import deque

class SafetyShield:
    def __init__(self, action_catalog, max_actions_per_tick=3, flap_window=5):
        self.action_catalog = {a['name']: a for a in action_catalog['actions']}
        self.max_actions = max_actions_per_tick
        self.flap_window = flap_window
        self.action_history = deque(maxlen=flap_window)

    def validate(self, proposals: list, fsm_state: str, bios_ok: bool) -> list:
        if fsm_state == "ERROR_STATE" or not bios_ok:
            return []

        filtered = []
        for p in proposals:
            if len(p.proposed_actions) > self.max_actions:
                continue
            valid = True
            for cmd in p.proposed_actions:
                if not self._validate_action(cmd):
                    valid = False
                    break
            if valid and not self._is_flapping(p):
                filtered.append(p)
                self.action_history.append(p.proposed_actions[0].name if p.proposed_actions else None)
        return filtered

    def _validate_action(self, cmd: ActuatorCommand) -> bool:
        meta = self.action_catalog.get(cmd.name)
        if not meta:
            return False
        for k, v in cmd.params.items():
            vmin, vmax = meta['params'].get(k, (None, None))
            if vmin is not None and not (vmin <= v <= vmax):
                return False
        return True

    def _is_flapping(self, proposal):
        if not self.action_history:
            return False
        last_action = self.action_history[-1]
        current_action = proposal.proposed_actions[0].name if proposal.proposed_actions else None
        return last_action == current_action and len([a for a in self.action_history if a == current_action]) >= 2
