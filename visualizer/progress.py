from typing import Iterable, Optional
from tqdm import tqdm


class ProgressBar(tqdm):
    """Wraps tqdm progress bar."""

    def __init__(
        self,
        iterable: Iterable,
        total: int,
        unit: str = "batch",
        fmt: bool = True,
    ):
        if fmt:
            bar_format = "{desc:<20.20}{percentage:6.0f}%|{bar:16}{r_bar}"
        else:
            bar_format = None
        # ascii_symbol = None if train else " ="
        super(ProgressBar, self).__init__(
            iterable=iterable,
            unit="batch",
            leave=True,
            # ascii=ascii_symbol,
            bar_format=bar_format,
            total=total,
            ncols=140,
        )
