import time
import json
import os
from typing import Dict, Any, Optional, List, Tuple


class MemoryEntry:
    def __init__(self, key: str, value: Any, trust: float = 0.95, step: int = 0):
        self.key = key
        self.value = value
        self.hits: int = 1  # phi_hits (Frequency)
        self.trust: float = float(trust)  # tau_trust (Authority)
        self.last_accessed_step: int = step  # Last accessed step tick
        self.last_accessed_time: float = time.time()  # Last accessed timestamp (seconds)
        self.created_step: int = step

    def get_priority_score(self, current_step: int) -> float:
        """
        Calculates Priority Score using Option B Domain-Weighted Fusion formula:
        Priority Score = (phi_hits * tau_trust) / (delta_t_last_accessed + 1.0)
        """
        delta_t = max(0, current_step - self.last_accessed_step)
        priority_score = (self.hits * self.trust) / (float(delta_t) + 1.0)
        return priority_score

    def to_dict(self, current_step: int) -> Dict[str, Any]:
        delta_t = max(0, current_step - self.last_accessed_step)
        score = self.get_priority_score(current_step)
        return {
            "key": self.key,
            "value": self.value,
            "hits": self.hits,
            "trust": self.trust,
            "last_accessed_step": self.last_accessed_step,
            "delta_t": delta_t,
            "priority_score": round(score, 4)
        }


class MedStreamMem:
    """
    Bounded Memory Buffer using Domain-Weighted Fusion (Frequency + Recency + Trust) formula.
    """
    def __init__(self, capacity: int = 50, default_trust: float = 0.95):
        self.capacity: int = max(1, capacity)
        self.default_trust: float = default_trust
        self.buffer: Dict[str, MemoryEntry] = {}
        self.current_step: int = 0
        self.eviction_history: List[Dict[str, Any]] = []

    def get(self, key: str) -> Optional[Any]:
        self.current_step += 1
        if key in self.buffer:
            entry = self.buffer[key]
            entry.hits += 1
            entry.last_accessed_step = self.current_step
            entry.last_accessed_time = time.time()
            return entry.value
        return None

    def put(self, key: str, value: Any, trust: Optional[float] = None) -> Tuple[bool, Optional[str]]:
        self.current_step += 1
        entry_trust = trust if trust is not None else self.default_trust

        # Key already exists: update value, increment hits, refresh timestamp
        if key in self.buffer:
            entry = self.buffer[key]
            entry.value = value
            entry.hits += 1
            entry.trust = entry_trust
            entry.last_accessed_step = self.current_step
            entry.last_accessed_time = time.time()
            return False, None

        evicted_key = None
        # Memory is full: evict entry with the lowest Priority Score
        if len(self.buffer) >= self.capacity:
            evicted_key = self.evict_lowest()

        # Insert new entry
        new_entry = MemoryEntry(key=key, value=value, trust=entry_trust, step=self.current_step)
        self.buffer[key] = new_entry
        return True, evicted_key

    def evict_lowest(self) -> str:
        """
        Finds and removes the memory entry with the lowest Priority Score.
        """
        if not self.buffer:
            raise KeyError("Cannot evict from empty memory buffer.")

        lowest_key = None
        lowest_score = float('inf')

        for key, entry in self.buffer.items():
            score = entry.get_priority_score(self.current_step)
            if score < lowest_score:
                lowest_score = score
                lowest_key = key

        evicted_entry = self.buffer.pop(lowest_key)
        record = {
            "evicted_key": lowest_key,
            "evicted_score": round(lowest_score, 4),
            "hits": evicted_entry.hits,
            "trust": evicted_entry.trust,
            "delta_t": self.current_step - evicted_entry.last_accessed_step,
            "step": self.current_step
        }
        self.eviction_history.append(record)
        return lowest_key

    def get_all_entries(self) -> List[Dict[str, Any]]:
        entries = [entry.to_dict(self.current_step) for entry in self.buffer.values()]
        # Sort by priority score descending
        entries.sort(key=lambda x: x["priority_score"], reverse=True)
        return entries

    def get_telemetry(self) -> Dict[str, Any]:
        return {
            "capacity": self.capacity,
            "current_size": len(self.buffer),
            "current_step": self.current_step,
            "total_evictions": len(self.eviction_history),
            "entries": self.get_all_entries(),
            "eviction_history": self.eviction_history[-10:]  # last 10 evictions
        }

    def save_snapshot(self, filepath: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        data = {
            "capacity": self.capacity,
            "default_trust": self.default_trust,
            "current_step": self.current_step,
            "eviction_history": self.eviction_history,
            "entries": [
                {
                    "key": entry.key,
                    "value": entry.value,
                    "hits": entry.hits,
                    "trust": entry.trust,
                    "last_accessed_step": entry.last_accessed_step,
                    "created_step": entry.created_step
                }
                for entry in self.buffer.values()
            ]
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load_snapshot(self, filepath: str) -> bool:
        if not os.path.exists(filepath):
            return False
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.capacity = data.get("capacity", self.capacity)
            self.default_trust = data.get("default_trust", self.default_trust)
            self.current_step = data.get("current_step", 0)
            self.eviction_history = data.get("eviction_history", [])
            self.buffer.clear()
            for e_data in data.get("entries", []):
                entry = MemoryEntry(
                    key=e_data["key"],
                    value=e_data["value"],
                    trust=e_data["trust"],
                    step=e_data["last_accessed_step"]
                )
                entry.hits = e_data["hits"]
                entry.created_step = e_data.get("created_step", entry.last_accessed_step)
                self.buffer[entry.key] = entry
            return True
        except Exception as e:
            print(f"Error loading memory snapshot from {filepath}: {e}")
            return False
