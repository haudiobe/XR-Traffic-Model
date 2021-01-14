# XR-Traffic-Model

## Installation

> **python 3.6.9 or above is needed**

**get the code**
```
git clone https://github.com/haudiobe/XR-Traffic-Model.git
git checkout dev 
cd ./XR-Traffic-Model
```

**create a virtual python environment**
```
python3 -m venv ./myvenv
```

**activate the virtual env**
> Windows/OSX: look up inside the ./myvenv directory for the relevant activation script.
```
./myvenv/bin/activate 
```

**install dependencies the dependencies**
```
(myvenv) pip install -r ./requirements.txt
```

## Usage

The input/output filenames are prefixed automatically with user ids. Currently, you must run these commands once per user.

**Generate S-Trace from V-Trace for user id #3**
```
(myvenv) python ./xrtm_encoder.py -c ./samples/encoder.cfg.json --user_id 3
```
outputs :
* `./samples_results/S-Trace[3].csv` S-Trace file for all buffers
* `./samples_results/S-Trace[3].frames/` directory containing all traces


**Generate P-Trace from S-Trace**
```
(myvenv) python ./xrtm_packetizer.py -c ./samples/packetizer.cfg.json --user_id 3
```
outputs :
* `./samples_results/P-Trace[3].csv` P-Trace file for all buffers



