# XR-Traffic-Model


## installation

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
> windows/osx: look up the ./myvenv for the relevant actiavtion script
```
./myvenv/bin/activate 
```

**install dependencies the dependencies**
```
(myvenv) pip install -r ./requirements.txt
```

## Usage

1. generate S trace from V trace
```
(myvenv) python ./xrtm_encoder.py -c ./encoder.cfg -v ./myfile.vtrace.csv 
```
outputs `./myfile.strace.csv`

2. generate P trace from S trace
```
(myvenv) python ./xrtm_packetizer.py -s ./myfile.strace.csv 
```
outputs `./myfile.ptrace.csv`

3. reconstruct from P' trace
```
(myvenv) python ./xrtm_packetizer.py -p ./myfile.ptrace.csv -s ./myfile.strace.csv 
```
outputs `./myfile.strace-rx.csv`
***NOTE !*** currently, only S'Trace is generated. V'Trace implementation and feedback implementation is under active development. 

### CSV format

see `./sample.vtrace.csv` for sample data.
***NOTE !*** the csv property names have are not yet updated.

### configuration

#### strace generation
see `./encoder.cfg`

#### ptrace generation
***TODO*** (network delay, ...)

#### reconstruction
***TODO***
