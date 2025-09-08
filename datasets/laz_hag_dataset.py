import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
import laspy

class LazHAGDataset(Dataset):
    """Dataset for reading LAS/LAZ files with optional HAG field.

    Parameters
    ----------
    root : str
        Root directory containing the ``train``, ``val`` or ``test`` folders.
    split : str
        One of ``'train'``, ``'val'`` or ``'test'`` specifying which sub-folder
        to read from.
    with_hag : bool, optional
        If ``True`` the dataset expects the input files to contain a ``HAG``
        (height above ground) dimension and will return point/HAG pairs.  If
        ``False`` only the XYZ coordinates are returned.  This is useful for
        the test phase where ground truth HAG values are not available.
    """

    def __init__(self, root: str, split: str = "train", with_hag: bool = True):
        super().__init__()
        self.with_hag = with_hag
        split_dir = os.path.join(root, split)
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(f"Split directory '{split_dir}' was not found")
        # Accept both LAS and LAZ extensions
        files = sorted(glob.glob(os.path.join(split_dir, "*.la[sz]")))
        if len(files) == 0:
            raise FileNotFoundError(f"No LAS/LAZ files found in '{split_dir}'")

        pts = []
        hags = []
        for fp in files:
            las = laspy.read(fp)
            xyz = np.vstack((las.x, las.y, las.z)).T.astype(np.float32)
            pts.append(xyz)
            if self.with_hag:
                if "HAG" in las.point_format.extra_dimension_names:
                    hag = las["HAG"].astype(np.float32)
                elif "hag" in las.point_format.extra_dimension_names:
                    hag = las["hag"].astype(np.float32)
                else:
                    raise ValueError(f"HAG dimension not found in file '{fp}'")
                hags.append(hag.reshape(-1, 1))
        self.xyz = np.concatenate(pts, axis=0)
        if self.with_hag:
            self.hag = np.concatenate(hags, axis=0)

    def __len__(self):
        return self.xyz.shape[0]

    def __getitem__(self, idx):
        xyz = torch.from_numpy(self.xyz[idx])
        if self.with_hag:
            hag = torch.from_numpy(self.hag[idx])
            return {"xyz": xyz, "hag": hag}
        return {"xyz": xyz}
