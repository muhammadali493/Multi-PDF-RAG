
import os, json
from typing import Set

class ProcessedHashesRepo:
    def __init__(self, path: str) -> None:
        self.path = path
        os.makedirs(os.path.dirname(self.path), exist_ok=True)

    def load(self) -> Set[str]:
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    return set(json.load(f))
            except Exception:
                return set()
        return set()

    def save(self, hashes: set[str]) -> None:
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(list(hashes), f)
