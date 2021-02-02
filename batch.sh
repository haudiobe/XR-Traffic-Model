#!/bin/sh

./batch.py -s ./VR/vr2-1.encoder.json -p ./VR/vr2-1.packetizer.json -o ./VR/vr2-1
./batch.py -s ./VR/vr2-2.encoder.json -p ./VR/vr2-2.packetizer.json -o ./VR/vr2-2

./batch.py -s ./VR/vr2-3.encoder.json -p ./VR/vr2-3.packetizer.json -o ./VR/vr2-3
./batch.py -s ./VR/vr2-4.encoder.json -p ./VR/vr2-4.packetizer.json -o ./VR/vr2-4

./batch.py -s ./VR/vr2-5.encoder.json -p ./VR/vr2-5.packetizer.json -o ./VR/vr2-5 
./batch.py -s ./VR/vr2-6.encoder.json -p ./VR/vr2-6.packetizer.json -o ./VR/vr2-6 

./batch.py -s ./VR/vr2-7.encoder.json -p ./VR/vr2-7.packetizer.json -o ./VR/vr2-7 
./batch.py -s ./VR/vr2-8.encoder.json -p ./VR/vr2-8.packetizer.json -o ./VR/vr2-8 
