# UAV-RF-Fingerprint-
This repository includes data of micro UAV RF signal received by USPR B210, matlab code for decomposing the raw signal data into few IMFs with EMD and EEMD. STFT is used in obtaining the features of each IMF on spectrum. Colab code includes multiple ML classifiers used in categorizing the features of RF signal in UAV detection.  
Matlab data file after extracting time-frequency features by STFT: https://drive.google.com/drive/folders/1BECYsFS-IcnBZVThTevdL6yyWPvAzuMp?usp=sharing

Raw cf.32 signal data received by USRP b210: https://drive.google.com/drive/folders/1LETvV20HV25hFgVFArUqHSspYAxhz7Z4?usp=sharing

Notation explaination: v-video (downlink) signal, c-control (uplink) signal, 1,2-number of unknown UAV in test field, straight-surround- flying trajectory, RX#-receiver channel 
