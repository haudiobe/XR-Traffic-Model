# XR-Traffic-Model

## Tools Used 
* python3.6
* asyncio python

## Running
1. Run the multiple_servers.py
2. Run the multiple_clients.py

The client server interaction is as shown below. It contains control and data communication parts. Control part is designed to include the feedback information from the client whereas the data part is designed to send the read information to client. The messages in the control part is just exemplary and after the traffic modelling and feedback API is designed, this part should be changed to postprocess the feedback.

![client_server_chart](https://user-images.githubusercontent.com/35132231/85281440-46879a80-b48a-11ea-8c85-38d479012d00.png)

After the above steps, .dat file is read frame by frame by multiple_servers.py and sent to the client. CSV file reading is also possible and it can be enabled by uncommenting the commented section in multiple_servers.py. By default, only the .dat file is read and sent.

__NOTE:__ The given .dat files seem to be produced with [x265 v2.3](https://github.com/videolan/x265/releases/tag/2.3). Therefore, the reading of the .dat file is done according to this version. Please note that newer versions of x265 seem to have more information dumped to .dat file (including motion vectors, etc.). If the used version changes to generate the .dat file, FrameInformation class in multiple_servers.py should be updated accordingly.

__NOTE:__ Currently, only one .dat file is read and sent from the server. This can be made into a loop quite easily.
