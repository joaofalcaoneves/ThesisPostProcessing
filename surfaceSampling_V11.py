#!/usr/bin/env python3

################################################################################
# AUTHOR: João Falcão Neves
# VERSION: 1.110
# DATE: 02/01/2020
# BUGS:
# 1) Some timesteps disappear due to accuracy of Z location of isosurface (not
# zero) - solved (kind of).
################################################################################
# Description:
#  Python3 script to get the IsoSurface at distances X and Z from several time
#  step folders into a IsoSurfaceX#.#m_Z#.#m.csv file.
#
# Instructions:
#  Install "pandas" module:
#  < sudo apt-get install python3-pandas >
#
#  To make the script executable go to file directory and type in the terminal:
#    < chmod u+x surfaceSamplingV1.100.py >
#  Otherwise type, each time:
#    < python3 surfaceSamplingV1.100.py >
################################################################################
# Import modules:
import os
import csv
import glob
import pandas as pd
import numpy as np

################################################################################
################################################################################
# Clock
from datetime import datetime

start_time = datetime.now()
################################################################################
# Numeric values, in [m], for (X(i),Z(i)) location of IsoSurface.
XValues = [474, 500, 726]  # Must be in a cell vertice ending with .0 in X.
ZValues = [0, 0, 0]  # Must be in a cell vertice

# Convert to numpy array and float
ArrayX = np.array(XValues, dtype=np.float64)
ArrayZ = np.array(ZValues, dtype=np.float64)

################################################################################
# Main folder path
mainPath = '/home/joaofn/OpenFOAM/OpenFOAM-2.4.0/JoaoFN/differentTimeScheme/CrankNicolson/CrankNicolson_08/H3/'
# List all file directories
# pathToTimeStep = glob.glob(mainPath + 'postProcessing/surfaceSampling/*')

# Iterate for (X(i),Z(i)) location of IsoSurface.
for i in range(len(ArrayX)):
    x = ArrayX[i]
    z = ArrayZ[i]

    # Open/Create new file, to write Y values and time of the isosurface at
    # XValues and ZValues location, as an csv file
    with open(mainPath + 'postProcessing/surfaceSampling/IsoSurface_X' + str(x) + 'm_Z' + str(z) + 'm.csv', 'w',
              newline='') as csvfile:

        fieldnames = ['Time', 'X', 'Y', 'Z', 'alphaWater']

        thewriter = csv.DictWriter(csvfile, dialect='excel', fieldnames=fieldnames)
        thewriter.writeheader()

        # for loop to open all files within different timeStep directories
        for pathToTimeStep in glob.glob(mainPath + 'postProcessing/surfaceSampling/*'):

            wrongPath = glob.glob(mainPath + 'postProcessing/surfaceSampling/[A-z]*')  # '.csv')
            # skip error if goes into IsoSurface(...).csv as a directory
            if pathToTimeStep in wrongPath:
                continue
            else:
                timeStep = os.path.basename(pathToTimeStep)

                # next two lines open file starting on 3rd row and add header to
                # help filter by column name, reads data type as float64 (maybe
                # too much memory used unnecessary)
                headerNames = ['X', 'Y', 'Z', 'alphaWater']
                file = pd.read_csv(pathToTimeStep + '/alpha.phase1_alpha0.50.raw', delim_whitespace=True,
                                   skiprows=[0, 1], header=None, names=headerNames, dtype=float)

                # Round up to maximum 4 decimal places to avoid skiping
                # timesteps with coordinate values like 1E-8 as 0.
                # Files "alpha.phase1_alpha0.50.raw" give 3 values for Z per X value:
                # [X1 Y Z1=0] | [X1 Y Z2=0.5] | [X1 Y Z3=0.25], !! sometimes there are duplicate lines !!
                file.Z = file.Z.round(4)

                # Filter all unwanted values
                fileXFiltered = file[file.X == x]
                fileXZFiltered = fileXFiltered[fileXFiltered.Z == z]

                # Insert timeStep column
                fileXZFiltered.insert(0, 'timeStep', float(timeStep))

                # Convert from dataframe to list
                fileXZFiltered = fileXZFiltered.values.tolist()

                # Write to csv file
                thewriter = csv.writer(csvfile, delimiter=',', dialect='excel', escapechar=' ', quoting=csv.QUOTE_NONE)
                thewriter.writerows(fileXZFiltered)

    fileToSort = pd.read_csv(mainPath + 'postProcessing/surfaceSampling/IsoSurface_X' + str(x)
                             + 'm_Z' + str(z) + 'm.csv')
    sortedWave = fileToSort.sort_values(by='Time')
    sortedWave.to_csv(mainPath + 'postProcessing/surfaceSampling/IsoSurface_X' + str(x)
                      + 'm_Z' + str(z) + 'm.csv', index=False)

################################################################################
################################################################################
print('Finnished!')
end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))
