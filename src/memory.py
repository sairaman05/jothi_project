import time
import json
import os
import sys
from typing import Dict, Any, Optional, List, Tuple


class MemoryEntry:
    def __init__(self, key: str, value: Any, trust: float = 0.95, step: int = 0):
        self.key: str = key
        self.value: Any = value
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

    def get_estimated_bytes(self) -> int:
        """
        Estimates the memory footprint in bytes for this entry.
        Includes key, value serialized size, plus Python dict/object overhead.
        """
        size = sys.getsizeof(self) + sys.getsizeof(self.key)
        try:
            val_str = json.dumps(self.value, default=str)
            size += len(val_str.encode('utf-8'))
        except Exception:
            size += sys.getsizeof(self.value)
        # Add baseline metadata overhead (references, timestamps, attributes)
        size += 256
        return size

    def to_dict(self, current_step: int) -> Dict[str, Any]:
        delta_t = max(0, current_step - self.last_accessed_step)
        score = self.get_priority_score(current_step)
        return {
            "key": self.key,
            "value": self.value,
            "hits": self.hits,
            "trust": round(self.trust, 4),
            "last_accessed_step": self.last_accessed_step,
            "created_step": self.created_step,
            "delta_t": delta_t,
            "priority_score": round(score, 4),
            "size_bytes": self.get_estimated_bytes()
        }


class BaseMemory:
    """
    Base memory interface and telemetry tracker for all comparative memory classes.
    """
    def __init__(self, capacity: Optional[int] = 50, default_trust: float = 0.95, name: str = "BaseMemory"):
        self.name: str = name
        self.capacity: Optional[int] = capacity
        self.default_trust: float = default_trust
        self.buffer: Dict[str, MemoryEntry] = {}
        self.current_step: int = 0
        self.eviction_history: List[Dict[str, Any]] = []
        
        # Telemetry counters
        self.total_lookups: int = 0
        self.cache_hits: int = 0
        self.cache_misses: int = 0
        self.hit_latencies: List[float] = []
        self.miss_latencies: List[float] = []

    def get(self, key: str, latency: float = 0.001) -> Optional[Any]:
        self.current_step += 1
        self.total_lookups += 1
        if key in self.buffer:
            entry = self.buffer[key]
            entry.hits += 1
            entry.last_accessed_step = self.current_step
            entry.last_accessed_time = time.time()
            self.cache_hits += 1
            self.hit_latencies.append(latency)
            return entry.value
        self.cache_misses += 1
        return None

    def record_miss_latency(self, latency: float):
        """Records latency when a cache miss occurred and LLM pipeline was executed."""
        self.miss_latencies.append(latency)

    def put(self, key: str, value: Any, trust: Optional[float] = None) -> Tuple[bool, Optional[str]]:
        raise NotImplementedError

    def evict(self) -> str:
        raise NotImplementedError

    def get_cache_hit_ratio(self) -> float:
        total = self.cache_hits + self.cache_misses
        if total == 0:
            return 0.0
        return (self.cache_hits / total) * 100.0

    def get_average_latency(self) -> float:
        """
        L_bar = CHR * L_hit + (1 - CHR) * L_miss
        Defaults: hit ~ 0.001s, miss ~ 2.5s (or empirical measurements)
        """
        chr_ratio = self.get_cache_hit_ratio() / 100.0
        avg_hit = (sum(self.hit_latencies) / len(self.hit_latencies)) if self.hit_latencies else 0.001
        avg_miss = (sum(self.miss_latencies) / len(self.miss_latencies)) if self.miss_latencies else 12.4
        return (chr_ratio * avg_hit) + ((1.0 - chr_ratio) * avg_miss)

    def get_high_trust_retention_ratio(self, threshold: float = 0.90) -> float:
        """
        HTRR (%) = (Sum I(tau_e >= threshold) / current_size) * 100%
        """
        if not self.buffer:
            return 0.0
        high_trust_count = sum(1 for e in self.buffer.values() if e.trust >= threshold)
        return (high_trust_count / len(self.buffer)) * 100.0

    def get_memory_usage_bytes(self) -> int:
        """
        Computes total estimated RAM bytes occupied by current memory buffer.
        """
        base_size = sys.getsizeof(self.buffer)
        entries_size = sum(entry.get_estimated_bytes() for entry in self.buffer.values())
        return base_size + entries_size

    def get_memory_usage_mb(self) -> float:
        """
        Returns estimated RAM in Megabytes.
        """
        return round(self.get_memory_usage_bytes() / (1024.0 * 1024.0), 4)

    def get_all_entries(self) -> List[Dict[str, Any]]:
        entries = [entry.to_dict(self.current_step) for entry in self.buffer.values()]
        entries.sort(key=lambda x: x["priority_score"], reverse=True)
        return entries

    def get_telemetry(self) -> Dict[str, Any]:
        cur_mb = self.get_memory_usage_mb()
        chr_pct = round(self.get_cache_hit_ratio(), 2)
        htrr_pct = round(self.get_high_trust_retention_ratio(), 2)
        avg_lat = round(self.get_average_latency(), 4)

        return {
            "name": self.name,
            "capacity": self.capacity if self.capacity is not None else "Unbounded",
            "current_size": len(self.buffer),
            "current_step": self.current_step,
            "total_lookups": self.total_lookups,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_ratio_pct": chr_pct,
            "high_trust_retention_pct": htrr_pct,
            "average_latency_sec": avg_lat,
            "memory_usage_bytes": self.get_memory_usage_bytes(),
            "memory_usage_mb": cur_mb,
            "total_evictions": len(self.eviction_history),
            "entries": self.get_all_entries(),
            "eviction_history": self.eviction_history[-10:]
        }

    def save_snapshot(self, filepath: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        data = {
            "name": self.name,
            "capacity": self.capacity,
            "default_trust": self.default_trust,
            "current_step": self.current_step,
            "total_lookups": self.total_lookups,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
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
            self.total_lookups = data.get("total_lookups", 0)
            self.cache_hits = data.get("cache_hits", 0)
            self.cache_misses = data.get("cache_misses", 0)
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


class UnboundedMemory(BaseMemory):
    """
    Baseline 1: Unbounded Growth Memory (tracks Memory Before Value).
    Memory buffer grows linearly O(N) with query volume, never evicting.
    """
    def __init__(self, default_trust: float = 0.95):
        super().__init__(capacity=None, default_trust=default_trust, name="UnboundedMemory")

    def put(self, key: str, value: Any, trust: Optional[float] = None) -> Tuple[bool, Optional[str]]:
        self.current_step += 1
        entry_trust = trust if trust is not None else self.default_trust

        if key in self.buffer:
            entry = self.buffer[key]
            entry.value = value
            entry.hits += 1
            entry.trust = entry_trust
            entry.last_accessed_step = self.current_step
            entry.last_accessed_time = time.time()
            return False, None

        # Unbounded: always insert, never evict
        new_entry = MemoryEntry(key=key, value=value, trust=entry_trust, step=self.current_step)
        self.buffer[key] = new_entry
        return True, None

    def evict(self) -> str:
        raise NotImplementedError("UnboundedMemory does not evict entries.")


class LRUMemory(BaseMemory):
    """
    Baseline 2: Least Recently Used (LRU) Memory.
    Fixed capacity K. Evicts the entry with the oldest last_accessed_step (largest delta_t).
    Vulnerability: recent low-authority queries (tau = 0.50) evict authoritative clinical guidelines (tau = 0.95).
    """
    def __init__(self, capacity: int = 50, default_trust: float = 0.95):
        super().__init__(capacity=max(1, capacity), default_trust=default_trust, name="LRUMemory")

    def put(self, key: str, value: Any, trust: Optional[float] = None) -> Tuple[bool, Optional[str]]:
        self.current_step += 1
        entry_trust = trust if trust is not None else self.default_trust

        if key in self.buffer:
            entry = self.buffer[key]
            entry.value = value
            entry.hits += 1
            entry.trust = entry_trust
            entry.last_accessed_step = self.current_step
            entry.last_accessed_time = time.time()
            return False, None

        evicted_key = None
        if len(self.buffer) >= self.capacity:
            evicted_key = self.evict()

        new_entry = MemoryEntry(key=key, value=value, trust=entry_trust, step=self.current_step)
        self.buffer[key] = new_entry
        return True, evicted_key

    def evict(self) -> str:
        if not self.buffer:
            raise KeyError("Cannot evict from empty LRU memory buffer.")

        # Evict entry with lowest last_accessed_step (oldest access)
        oldest_key = min(self.buffer.keys(), key=lambda k: self.buffer[k].last_accessed_step)
        evicted_entry = self.buffer.pop(oldest_key)
        record = {
            "evicted_key": oldest_key,
            "eviction_policy": "LRU",
            "hits": evicted_entry.hits,
            "trust": evicted_entry.trust,
            "delta_t": self.current_step - evicted_entry.last_accessed_step,
            "step": self.current_step
        }
        self.eviction_history.append(record)
        return oldest_key


class LFUMemory(BaseMemory):
    """
    Baseline 3: Least Frequently Used (LFU) Memory.
    Fixed capacity K. Evicts the entry with the lowest hit count (phi_hits).
    Flaw: ignores recency decay (delta_t); stale queries stay forever while new guidelines with hit=1 are evicted.
    """
    def __init__(self, capacity: int = 50, default_trust: float = 0.95):
        super().__init__(capacity=max(1, capacity), default_trust=default_trust, name="LFUMemory")

    def put(self, key: str, value: Any, trust: Optional[float] = None) -> Tuple[bool, Optional[str]]:
        self.current_step += 1
        entry_trust = trust if trust is not None else self.default_trust

        if key in self.buffer:
            entry = self.buffer[key]
            entry.value = value
            entry.hits += 1
            entry.trust = entry_trust
            entry.last_accessed_step = self.current_step
            entry.last_accessed_time = time.time()
            return False, None

        evicted_key = None
        if len(self.buffer) >= self.capacity:
            evicted_key = self.evict()

        new_entry = MemoryEntry(key=key, value=value, trust=entry_trust, step=self.current_step)
        self.buffer[key] = new_entry
        return True, evicted_key

    def evict(self) -> str:
        if not self.buffer:
            raise KeyError("Cannot evict from empty LFU memory buffer.")

        # Evict entry with lowest hit count (phi_hits). Tiebreaker: oldest last_accessed_step
        lowest_key = min(
            self.buffer.keys(),
            key=lambda k: (self.buffer[k].hits, self.buffer[k].last_accessed_step)
        )
        evicted_entry = self.buffer.pop(lowest_key)
        record = {
            "evicted_key": lowest_key,
            "eviction_policy": "LFU",
            "hits": evicted_entry.hits,
            "trust": evicted_entry.trust,
            "delta_t": self.current_step - evicted_entry.last_accessed_step,
            "step": self.current_step
        }
        self.eviction_history.append(record)
        return lowest_key


class MedStreamMem(BaseMemory):
    """
    Option B: Domain-Weighted Fusion Bounded Memory.
    Fixed capacity K (tracks Memory Optimized Value).
    Priority Score = (phi_hits * tau_trust) / (delta_t + 1.0)
    Evicts the entry with the lowest Priority Score, preserving high-trust clinical guidelines and popular queries.
    """
    def __init__(self, capacity: int = 50, default_trust: float = 0.95):
        super().__init__(capacity=max(1, capacity), default_trust=default_trust, name="MedStreamMem")

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

    def evict(self) -> str:
        return self.evict_lowest()

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


def calculate_memory_reduction(ram_before: float, ram_optimized: float) -> float:
    """
    MR % = (1 - RAM_Optimized / RAM_Before) * 100%
    """
    if ram_before <= 0:
        return 0.0
    reduction = (1.0 - (ram_optimized / ram_before)) * 100.0
    return round(max(0.0, min(100.0, reduction)), 2)


def create_memory(memory_type: str = "medstreammem", capacity: int = 50, default_trust: float = 0.95) -> BaseMemory:
    """
    Factory function to instantiate any of the 4 comparative memory classes.
    """
    m_type = memory_type.lower().strip()
    if m_type in ["unbounded", "unboundedmemory", "before"]:
        return UnboundedMemory(default_trust=default_trust)
    elif m_type in ["lru", "lrumemory"]:
        return LRUMemory(capacity=capacity, default_trust=default_trust)
    elif m_type in ["lfu", "lfumemory"]:
        return LFUMemory(capacity=capacity, default_trust=default_trust)
    elif m_type in ["medstreammem", "domain_weighted", "optimized", "after"]:
        return MedStreamMem(capacity=capacity, default_trust=default_trust)
    else:
        raise ValueError(f"Unknown memory type '{memory_type}'. Expected: 'unbounded', 'lru', 'lfu', 'medstreammem'.")
