import os
import pandas as pd
import ToolGrafic


def generateFilteredScoreChart(SIMULATION_PATH_GA1, dfDat, featureByType, allFeatureTypes, allFeatureTypes1, startColumn, endColumn, method ):
    #dfFilteredMethods = pd.DataFrame()
    nRows = dfDat.shape[0]
    lstType = []
    for row in range(0, nRows):
        nameKey = dfDat.ix[row, 1]
        numberKey = featureByType[nameKey]
        type = allFeatureTypes[numberKey]
        lstType.append(type)

    dfFilteredMethods = dfDat.ix[:, startColumn:endColumn]

    lstAvgFilteredMethods = list(dfFilteredMethods.mean(axis=1))
    dfFilteredByType = pd.concat([pd.DataFrame(lstType), pd.DataFrame(lstAvgFilteredMethods)], axis=1,
                                 ignore_index=True)
    dfFilteredByType.columns = ['FeatureType', 'Score']
    dfFilteredByType = (dfFilteredByType.groupby(['FeatureType'], as_index=False).mean())
    dfFilteredByType1 = dfFilteredByType.sort_values(by=['Score'], ascending=[False])
    dfFilteredByType1 = dfFilteredByType1.reset_index(drop=True)
    dfFilteredByType1.to_csv(os.path.join(os.path.sep, SIMULATION_PATH_GA1, method+".csv"))
    colorByFeatureType = {1: '#92D050', 2: '#EEB5AA', 3: '#FFC000', 4: '#6699FF', 5: '#A03070'}
    ToolGrafic.saveFeatureImportanceScoreByTypeG(list(dfFilteredByType1.ix[:, 0]), list(dfFilteredByType1.ix[:, 1]),
                                                 SIMULATION_PATH_GA1, method, 4,
                                                 colorByFeatureType, allFeatureTypes1)

if __name__ == "__main__":
    ROOT_PATH           = '/home/edwin/Desktop'
    SIMULATION_NAME1    = 'IGHISTRUCTAL'
    SIMULATION_NAME2    = 'igHiStructFull'
    SIMULATION_PATH_GA  = os.path.join(os.path.sep, ROOT_PATH, SIMULATION_NAME1, SIMULATION_NAME2, 'GAIMPORTANCE')
    dfDat               = pd.read_csv(os.path.join(SIMULATION_PATH_GA, 'FinalConsensusFeatureImportance.csv'))
    featureByType       = {'PPHO': 1, 'PPHI': 1, 'HCA': 1, 'HCB': 1, 'HCN': 1,
                            'NH': 2, 'NB': 2, 'NE': 2, 'NG': 2, 'NI': 2, 'NT': 2, 'NS': 2, 'NO': 2,
                            'RGY': 3, 'AVCA': 3, 'AVCB': 3, 'AVPH': 3, 'AVPS': 3, 'NHBO': 3,
                            'ENER': 4, 'DOPE': 4, 'DFIRE': 4, 'GOAP': 4, 'RWDD': 4,
                            'RWOD': 4, 'PLGSC': 4, 'PMSUB': 4, 'FRST': 4, 'FRRAP': 4, 'FRSOL': 4, 'FRHYD': 4, 'FRTOR': 4,
                            'CHVOL': 3, 'ANUA1': 3, 'ANUA2': 3, 'ANUA3': 3, 'ANUA4': 3}
    allFeatureTypes = {1: 'Phychochemical', 2: 'Sec. Structure', 3: 'Struc. Features', 4: 'Energy'}
    allFeatureTypes1 = {'Phychochemical': 1, 'Sec. Structure': 2, 'Struc. Features': 3, 'Energy': 4}
    generateFilteredScoreChart( SIMULATION_PATH_GA, dfDat, featureByType, allFeatureTypes, allFeatureTypes1, 2, 10, 'FILTM_ByTYPE' )
    generateFilteredScoreChart( SIMULATION_PATH_GA, dfDat, featureByType, allFeatureTypes, allFeatureTypes1, 10, 12, 'SENM_ByTYPE')
    generateFilteredScoreChart( SIMULATION_PATH_GA, dfDat, featureByType, allFeatureTypes, allFeatureTypes1, 12, 13, 'GAM_ByTYPE')