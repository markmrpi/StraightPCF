import os
import glob
import argparse
import numpy as np
import torch
import laspy

from models.hag_predictor import HAGPredictor


def predict():
    parser = argparse.ArgumentParser(description="Predict HAG for LAZ files")
    parser.add_argument("--data-root", default="./data", help="Root directory containing test folder")
    parser.add_argument("--checkpoint", required=True, help="Path to trained model checkpoint")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--out-dir", default="./predictions", help="Directory to store predicted LAZ files")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HAGPredictor(hidden_dim=args.hidden_dim, num_layers=args.layers).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    test_dir = os.path.join(args.data_root, "test")
    os.makedirs(args.out_dir, exist_ok=True)

    for fp in sorted(glob.glob(os.path.join(test_dir, "*.la[sz]"))):
        las = laspy.read(fp)
        xyz = np.vstack((las.x, las.y, las.z)).T.astype(np.float32)
        xyz_tensor = torch.from_numpy(xyz).to(device)
        with torch.no_grad():
            pred_hag = model(xyz_tensor).cpu().numpy().squeeze()
        if "HAG" not in las.point_format.extra_dimension_names:
            las.add_extra_dim(laspy.ExtraBytesParams(name="HAG", type=np.float32))
        las["HAG"] = pred_hag.astype(np.float32)
        out_fp = os.path.join(args.out_dir, os.path.basename(fp))
        las.write(out_fp)
        print(f"Wrote predictions to {out_fp}")


if __name__ == "__main__":
    predict()
