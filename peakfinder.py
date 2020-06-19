#!/usr/bin/env python3

import tkinter
from tkinter import filedialog
import matplotlib.pyplot as plt
import pandas as pd
import scipy.signal

pathToMainFolder = tkinter.filedialog.askdirectory(initialdir='~/OpenFOAM/OpenFOAM-2.4.0/')
print("\nYou are working within the following directory:\n" + pathToMainFolder + "\n")

forceFile = pd.read_csv(tkinter.filedialog.askopenfile(initialdir=pathToMainFolder,
                                                       title="Select force file"))
waveFile = pd.read_csv(tkinter.filedialog.askopenfile(initialdir=pathToMainFolder,
                                                      title="Select wave elevation file"))

forcePeaks, _ = scipy.signal.find_peaks(forceFile['YPressureF'], height=0)
wavePeaks, _ = scipy.signal.find_peaks(waveFile['Y'], height=0)

peakaboof = forceFile.iloc[forcePeaks, :]  # Force peaks
peakaboow = waveFile.iloc[wavePeaks, :]  # Wave peaks
print(peakaboof)
print(peakaboow)

plt.figure(1)
plt.plot(peakaboof['Time'], peakaboof['YPressureF'], '+', label='YY pressure force peaks', color='red')
plt.plot(forceFile['Time'], forceFile['YPressureF'], label='YY pressure force', color='black')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2)

plt.figure(2)
plt.plot(peakaboow['Time'], peakaboow['Y'], '+', label='Water surface elevation peaks', color='red')
plt.plot(waveFile['Time'], waveFile['Y'], label='Water surface elevation', color='black')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2)
plt.show()

savePath = tkinter.filedialog.askdirectory(initialdir='~/OpenFOAM/OpenFOAM-2.4.0/',
                                           title='Dir to save peakboo.CSV files')
peakaboof.to_csv(savePath + 'peakaboof.csv', sep=',', header=True)
peakaboow.to_csv(savePath + 'peakaboow.csv', sep=',', header=True)
