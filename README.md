# 4D_Radar_MOT

The code for reproducing experiment results in the conference paper "Which Framework is Suitable for Online 3D Multi-Object Tracking for Autonomous Driving with Automotive 4D Imaging Radar?" in *35th IEEE Intelligent Vehicles Symposium (IV 2024)*, # Oral Presentation (Top 5%) #.

## Installation
- Clone the repository and enter the directory.
```
git clone https://github.com/dinggh0817/4D_Radar_MOT.git
cd 4D_Radar_MOT
```
- For conda users, create the python environment and install required packages.
```
conda create --name 4D_MOT python==3.8.16
conda activate 4D_MOT
pip install -r requirements.txt
```

## Run Evaluation
- Run `.py` scripts to evaluate different tracking algorithms, e.g.,
```
python RUN_SMURF_GNN_PMB.py
```
- Note: The GNN-PMB and GGIW-PMBM filters rely on the Murty algorithm. The prebulit `murty.dll` and `murty.so` library files are provided. Change the library directory in `mhtdaClink.py` if you want to run evaluations on Windows platform. Please refer to [fastmurty](https://github.com/motrom/fastmurty) for source code and further information of Murty.

## Acknowledgement
The code in this repository is developed based on the following projects:
- https://github.com/yuhsuansia/Extended-target-PMBM-and-PMB-filters
- https://github.com/JonathonLuiten/TrackEval
- https://github.com/tudelft-iv/view-of-delft-dataset
- https://github.com/motrom/fastmurty
