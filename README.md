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

Explicit configuration options need to be passed over as command line arguments. For the full list of options and their default values, just run:
```
(myvenv) python ./xrtm_encoder.py -v ./vtrace.csv -s ./strace.csv
(myvenv) python ./xrtm_packetizer.py -s ./strace.csv -p ./ptrace.csv
```

### CSV format

see `./sample.vtrace.csv` for sample data
the csv property names have are not yet updated

### configuration

see `./encoder.cfg`
