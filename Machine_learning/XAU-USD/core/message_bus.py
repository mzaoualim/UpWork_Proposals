import time
from typing import Dict, Any

class MessageBus:
    def __init__(self):
        self.seen_ids = {}

    def send_order(self, order: Dict[str, Any]) -> Dict[str, Any]:
        cid = order.get('client_order_id')
        valid_until = order.get('valid_until', None)
        now = time.time()
        if cid in self.seen_ids:
            entry = self.seen_ids[cid]
            if now <= entry.get('valid_until', 0):
                return {'status': 'rejected', 'reason': 'duplicate', 'client_order_id': cid}
        self.seen_ids[cid] = {'order': order, 'valid_until': now + float(valid_until) if valid_until else now + 60}
        return {'status': 'accepted', 'client_order_id': cid}

    def heartbeat(self):
        return {'ts': time.time(), 'status': 'alive'}
