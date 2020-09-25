# XR-Traffic-Model

## installation

Python 3.6.9 or above is required. To install dependencies, run :
```
pip install -r ./requirements.txt
```

## V-trace to S-trace model encoder

The model encoder takes in a csv file with V-trace as rows, and by default outputs another csv with S-trace as rows.

Explicit configuration options need to be passed over as command line arguments. For the full list of options and their default values, just run:
```
python ./xrtm_encoder.py --help
```

### V-trace format

see `./sample.vtrace.csv` for sample data

### Example usage

the default frame size (w)2048 * (h)2048. see next section for customization option details.

model encoder without error resilience, an I-slices are generated for V-traces with 100% intra ctus
```
python ./xrtm_encoder.py -i ./sample.vtrace.csv --plot
```

model encode w/ periodic slice refresh, 16 slices per frame, with a custom CRF adjustment
```
python ./xrtm_encoder.py -i ./sample.vtrace.csv -s=16 -e=2 --crf=25 -o ./strace_out.csv
```

model encode w/ feedback based error resilience, 32 slices per frame, no CRF adjustment
```
python ./xrtm_encoder.py -i ./sample.vtrace.csv -s=32 -e=3 ./strace_out.csv
```

model encode w/ periodic frame refresh, one I-frame every 60 frames, 8 slices per frame 
```
python ./xrtm_encoder.py -i ./sample.vtrace.csv -e=1 -s=8 -g=60 ./strace_out.csv
```


## Configuration / CLI arguments

### width / height 

`-W` or `--width`

`-H` or `--height`

Frame width/height is not specified as part of the V-trace model. Instead, you specify it explicitely. 
The default values are **`-W=2048`** and **`-H=2048`**


### Slices

`-s` or `--slices`

specifies how many slices (S-trace) per frame (V-trace) the model encoder must produce


the resulting **slice height must be a multiple of 64**. 

Slices are generated by dividing the frame height evenly vertically, and using the full frame width.


### **Error resilience modes**

`-e` or `--erm`

3 error resilience modes are implemented.
Error resilience modes manages the intra coding decision, overriding the picture coding type specified in V-trace data.

#### **DISABLED** `-e=0` or `--erm=0`

> in this mode, feedack is not currently taken into account.

This is the default mode. It runs the encoder without error resilience. Whenever a V-trace has 100% of intra ctus, the corresponding S-traces will be encoded as I-slices, otherwise they get encoded as P-slices.


#### **PERIODIC_FRAME** `-e=1` or `--erm=1`

> in this mode, feedack is not currently taken into account.

an intra frame is inserted every *n* frames.

how often intra frame must be inserted is specified using the `-g` or `--gop` option

```
python ./xrtm_encoder.py -i ./vtrace.csv -s=4 -e=1 -g=60
```

in this mode, feedack is not currently taken into account.


#### **PERIODIC_SLICE** `-e=2` or `--erm=2`

> in this mode, feedack is not currently taken into account.

Only the first frame will be a full intra. Then periodic intra refresh is performed through slices. 
The intra slice index incremented on each frame: `is_intra_slice = ((frame_poc % slices_per_frame) == slice_idx)`.

```
python ./xrtm_encoder.py -i ./vtrace.csv -s=4 -e=2
```


### Feedback modes

The CLI encoder tool `xrtm_encoder.py` implements a random feedback generator to emulate client feedback.

For *all feedback modes* listed below:
- the first frame gets encoded as Intra, then all subsequent frames get encoded as Inter, the encoder feedback status signals otherwise.
- before encoding a new frame (V-trace), the encoder checks for *full intra refresh* feedback and *bitrate control* feedback.

#### NACK feedback `-e=3` or `--erm=3`

NACK feedback signals the client could not decode a specific slice for a specific frame.
All encoded slices are considered referenceable by the encoder, unless an explicit NACK feedback was received.


#### ACK feedback `-e=4` or `--erm=4`

ACK feedback signals the client could successfully decode a specific slice for a specific frame.
All encoded slices are considered *non-referenceable* by the encoder, unless an explicit ACK feedback was received.




### **Rate control mode**

the model encoder currently only supports CRF

`--crf`

This option generates S-trace with adjusted bit size that takes into account the specified target CRF.
When unspecified, the CRF from V-trace is used and remains unchanged. 


### **Misc options**

`--log_level` 

default=0 - set log level. 0:CRITICAL, 1:INFO, 2:DEBUG


`--plot`

plot V-trace and S-trace stats using matplotlib