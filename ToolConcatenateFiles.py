import csv
import glob
import os
import pandas as pd
from cStringIO import StringIO
import  numpy as np


ROOT_PATH             = os.path.join( os.path.sep, 'home', 'edwin','DataToJoin' )
NEW_FULL_NAME         = 'lmdsFull.csv'
FEATURES_NAME         = ['pop_hidrofilico', 'pop_hidrofobico', 'hse_ca', 'hse_cb', 'hse_cn', 'numberH', 'numberB', 'numberE',
                         'numberG', 'numberI', 'numberT', 'numberS', 'numberO', 'radiusGyration_method1', 'avgCADistance', 'avgCBDistance', 'avgPhi', 'avgPsi',
                         'hydroBonds', 'energy', 'DOPE', 'DFIRE', 'GOAP', 'RWDD', 'RWOD', 'PQLGScore', 'PQMaxSubScore', 'FRSTScore', 'RAPDFScore',
                         'SolvScore', 'HydroScore', 'TorsionScore', 'VOLCHullSc', 'AvgNumAtom14', 'AvgNumAtom24', 'AvgNumAtom34', 'AvgNumAtom44', 'GDT-TS']


lstFilesToConcatenate = glob.glob( os.path.join( os.path.sep, ROOT_PATH,'*.txt' ) )


for dfFile in lstFilesToConcatenate:
     with open(dfFile, 'r') as in_file:
         stripped = (line.strip() for line in in_file)
         #lines = (line.split(",") for line in stripped if line)
         lines = (line.split(",") for line in stripped if line)
         with open(dfFile[0:len(dfFile)-3]+'1csv', 'w') as out_file:
             writer = csv.writer(out_file)
             writer.writerows(lines)
lstFilesToConcatenate = glob.glob( os.path.join( os.path.sep, ROOT_PATH,'*.1csv' ) )
for dfFile in lstFilesToConcatenate:
     with open(dfFile[0:len(dfFile)-4]+'csv', 'wb') as outcsv:
         writer = csv.writer(outcsv)
         writer.writerow(FEATURES_NAME)
         with open(dfFile, 'rb') as incsv:
             next(incsv)
             reader = csv.reader(incsv)
             writer.writerows(row for row in reader)
files = [file for file in glob.glob( os.path.join( os.path.sep, ROOT_PATH,'*.1csv' )) ]
for file in files:
    os.remove(file)
lstFilesToConcatenate = glob.glob( os.path.join(os.path.sep, ROOT_PATH, '*.csv') )
temDf = pd.DataFrame()
for dfFile in lstFilesToConcatenate:
     #df    = pd.read_csv(dfFile)
     df    = pd.read_csv(StringIO(''.join(l.replace(';', ',') for l in open(dfFile))),  names=FEATURES_NAME)
     df1 = df.ix[1:df.shape[0], :]
     try:
         df2 = df1.as_matrix().astype(np.float)
     except:
         m=1

     #df = pd.read_table(f, sep='-', index_col=0, header=None, names=['A', 'B', 'C'],
     #                      lineterminator='\n')

     temDf = pd.concat([temDf, df1], axis=0, ignore_index=True)
temFinal = temDf.as_matrix().astype(np.float)
#dfFinal = pd.DataFrame(temFinal, columns=FEATURES_NAME)
temDf.to_csv(os.path.join( os.path.sep, ROOT_PATH , NEW_FULL_NAME ), index=False)

df1 = pd.read_csv(os.path.join( os.path.sep, ROOT_PATH , NEW_FULL_NAME ), sep="," )
m =1