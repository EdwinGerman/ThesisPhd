import os
import pandas as pd
import numpy as np
import ToolGrafic
#ROOT_PATH           = '/home/edwin/SIMULATIONS'
ROOT_PATH           = '/home/edwin/Desktop'
SIMULATION_NAME1    = 'IG_STRUCTURAL_HIRES'
SIMULATION_NAME2    = 'igHiStructFull'
#DATABASENAME        = '1ctf_protein_features.csv'
SIMULATION_PATH_GA  = os.path.join(os.path.sep, ROOT_PATH, SIMULATION_NAME1, SIMULATION_NAME2, 'GAIMPORTANCE')

dfDat = pd.read_csv(os.path.join(SIMULATION_PATH_GA, 'Other_FeatureImportanceMethods.csv'))

featureName = np.array(['PPHO', 'PPHI', 'HCA', 'HCB', 'HCN', 'NH', 'NB', 'NE', 'NG', 'NI', 'NT', 'NS', 'NO', 'RGY',
                        'AVCA', 'AVCB', 'AVPH', 'AVPS', 'NHBO', 'ENER', 'DOPE', 'DFIRE', 'GOAP', 'RWDD',
                        'RWOD', 'PLGSC', 'PMSUB', 'FRST', 'FRRAP', 'FRSOL', 'FRHYD', 'FRTOR', 'CHVOL',
                        'ANUA1', 'ANUA2', 'ANUA3', 'ANUA4'])
featureByType = {'PPHO': 1, 'PPHI': 1, 'HCA': 1, 'HCB': 1, 'HCN': 1,
                 'NH': 2, 'NB': 2, 'NE': 2, 'NG': 2, 'NI': 2, 'NT': 2, 'NS': 2, 'NO': 2,
                 'RGY': 3, 'AVCA': 3, 'AVCB': 3, 'AVPH': 3, 'AVPS': 3, 'NHBO': 3,
                 'ENER': 4, 'DOPE': 4, 'DFIRE': 4, 'GOAP': 4, 'RWDD': 4,
                 'RWOD': 4, 'PLGSC': 4, 'PMSUB': 4, 'FRST': 4, 'FRRAP': 4, 'FRSOL': 4, 'FRHYD': 4, 'FRTOR': 4,
                 'CHVOL': 3, 'ANUA1': 3, 'ANUA2': 3, 'ANUA3': 3, 'ANUA4': 3}
colorByFeatureType = {1: '#92D050', 2: '#EEB5AA', 3: '#FFC000', 4: '#6699FF', 5: '#A03070'}

ToolGrafic.saveFeatureImportanceScoreByType(list(dfDat.ix[:,1]), list(dfDat.ix[:,10]),
                                                SIMULATION_PATH_GA, 'OM', len(featureName),
                                                colorByFeatureType, featureByType)
