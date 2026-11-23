import pickle
from pathlib import Path
from typing import Self


class SavableModel:
    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as file:
            pickle.dump(self, file)

    @classmethod
    def load(cls, path: str | Path) -> Self:
        with Path(path).open("rb") as file:
            model = pickle.load(file)
        if not isinstance(model, cls):
            msg = f"{path} does not contain a {cls.__name__}"
            raise TypeError(msg)
        return model
