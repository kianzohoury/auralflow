# Copyright (c) 2022 Kian Zohoury
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT
# This code is part of the auralflow project linked below.
# https://github.com/kianzohoury/auralflow.git

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
        desc: str = "",
        show_rate: bool = False,
    ):
        bar_format = f"{desc}: " if desc else ""
        bar_format += "{percentage:3.0f}%|{bar:12}|{n_fmt}/{total_fmt} "
        r_bar = "eta: {remaining}" + "{postfix}"
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


def create_progress_bar(
    iterable: Iterable,
    total: int,
    unit: str = "batch",
    fmt: bool = True,
    desc: str = "",
) -> ProgressBar:
    """Creates a progress bar."""
    return ProgressBar(
        iterable=iterable, total=total, unit=unit, fmt=fmt, desc=desc
    )
