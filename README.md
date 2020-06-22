# XR-Traffic-Model

## Tools Used 
* python3.6
* asyncio python

## Running
1. Run the multiple_servers.py
2. Run the multiple_clients.py

After the above steps, .dat file is read frame by frame by multiple_servers.py and sent to the client. CSV file reading is also possible and it can be enabled by uncommenting the commented section in multiple_servers.py. By default, only the .dat file is read and sent.

__NOTE:__ The given .dat files seem to be produced with [x265 v2.3](https://github.com/videolan/x265/releases/tag/2.3). Therefore, the reading of the .dat file is done according to this version. Please note that newer versions of x265 seem to have more information dumped to .dat file (including motion vectors, etc.). If the used version changes to generate the .dat file, FrameInformation class in multiple_servers.py should be updated accordingly.

__NOTE:__ Currently, only one .dat file is read and sent from the server. This can be made into a loop quite easily.
