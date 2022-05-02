from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm


class ProgressBar(tqdm):
    """Custom tqdm progress bar for training/validation loops.

    Args:
        dataloader (torch.utils.data.DataLoader): A dataloader object.
        total (int): Max number of iterations.
        train (bool): Displays training bar if True, or validation bar if False.
            Default: True.
    """

    def __init__(self, dataloader: DataLoader, total: int, train: bool = True):
        desc = None if train else f"Evaluating..."
        ascii_symbol = None if train else " ="
        bar_format = "{desc:<20.20}{percentage:6.0f}%|{bar:16}{r_bar}"
        super(ProgressBar, self).__init__(
            iterable=dataloader,
            desc=desc,
            unit="batch",
            leave=True,
            ascii=ascii_symbol,
            bar_format=bar_format,
            total=total,
            ncols=120,
        )
