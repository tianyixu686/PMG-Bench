from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List


class DatasetBuilder(ABC):
    @abstractmethod
    def list_target_users(self) -> List[str]:
        raise NotImplementedError

    @abstractmethod
    def build_train_manifest(self, user_id: str, output_root: Path) -> Path | None:
        raise NotImplementedError

    @abstractmethod
    def get_test_samples(self) -> List[Dict]:
        raise NotImplementedError
