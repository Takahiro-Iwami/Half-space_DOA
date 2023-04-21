# Direction-of-Arrival Estimation in Half-Space from Single Sample Array Snapshot

Authors: T. Iwami, K. Sawai, A. Omoto

This repository contains all code used in the paper "Direction-of-Arrival Estimation in Half-Space from Single Sample Array Snapshot".

## File Structure

- `Experiment-A.py`: Code of Experiment A
  - This code compares the DOA estimation performance of the MUSIC algorithm and the proposed method for a pulse-like sound field. It requires the `numpy`, `scipy`, and `matplotlib` libraries.
- `Experiment-B.py`: Code of Experiment B
  - This code compares the DOA estimation performance of the MUSIC algorithm and the proposed method for a complex sound field. It requires the `numpy`, `scipy`, and `matplotlib` libraries.
- `Experiment-C.py`: Code of Experiment C
  - This code compares the DOA estimation performance of the MUSIC algorithm and the proposed method for a complex field in which the DOAs change. It requires the `numpy`, `scipy`, `matplotlib`, and `tqdm` libraries.
- `Experiment-D.py`: Code of Experiment D
  - This code compares the DOA estimation performance of the normal method and the weighted method for a pulse-like sound field. It requires the `numpy`, `scipy`, and `matplotlib` libraries.
- `LICENSE`: License
- `README.md`: This file
