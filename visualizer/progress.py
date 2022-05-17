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
        desc: Optional[str] = None,
    ):
        bar_format = "{n_fmt}/{total_fmt:<10}|{bar:18}| "
        r_bar = "{elapsed}<{remaining}, {rate_fmt}{postfix}"
        bar_format += r_bar if fmt else "{rate_fmt}{postfix}"

        super(ProgressBar, self).__init__(
            iterable=iterable,
            unit=unit,
            leave=True,
            desc=desc,
            bar_format=bar_format,
            total=total,
            ncols=79,
        )
