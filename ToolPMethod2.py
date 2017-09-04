import copy
import multiprocessing
import numpy as np
import os
import pandas as pd
import scipy
import sys
from functools import partial
from scipy import stats

import sklearn
from sklearn import preprocessing
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler

import ToolGrafic


def getScaledDatabase( dfProProteina, trainFraction ):
    #Database            = pd.read_csv( databasePath, header=None )
    #must be filtered the features selected by the model
    dfProProteina       = sklearn.utils.shuffle(dfProProteina)
    dfProProteina       = dfProProteina.reset_index(drop=True)
    arrayDatabase       = dfProProteina.as_matrix()

    arrayDatabase       = arrayDatabase[1:arrayDatabase.shape[0], :]

    scaler              = preprocessing.StandardScaler().fit( arrayDatabase )

    scaled_database     = scaler.transform( arrayDatabase )

    examplesNumber      = scaled_database.shape[0]
    totalColumns        = scaled_database.shape[1]

    numTrainExamples    = int( examplesNumber * trainFraction )
    if numTrainExamples> examplesNumber:
        numTrainExamples = examplesNumber

    scaled_train       = scaled_database[0:numTrainExamples, :]
    scaled_test        = scaled_database[numTrainExamples:examplesNumber,:]

    x_Train            = scaled_train[:,0:totalColumns-1]
    y_Train            = scaled_train[:, totalColumns-1]

    x_Test             = scaled_test[:, 0:totalColumns-1]
    y_Test             = scaled_test[:, totalColumns-1]
    meanTarget         = scaler.mean_[totalColumns-1]
    stdTarget          = scaler.scale_[totalColumns-1]
    return x_Train, y_Train, x_Test, y_Test, meanTarget, stdTarget
def rmsCalculation( trueValue, predictedValue ):
    #x = np.array([2, 3, 1, 0])
    #y = np.array([1, 2, 1, 1])
    difference  = (trueValue - predictedValue)
    rmse        = difference * difference
    rmse        = rmse.sum(axis=0)
    n           = trueValue.shape[0]
    rmse        = pow( rmse / n, 0.5)
    return rmse
def getVarDesnormalization( arrVar, varMean, varStd):
    retunedValue = arrVar*varStd+varMean
    return retunedValue
def f( numberOfFeatures, dfDatabase, maxNumOfTrials,trainFactor, allExperiments ):
    ####Begin Process
    scipy.random.seed()
    x_Train, y_Train, x_Test, y_Test, meanTarget, stdTarget = getScaledDatabase(dfDatabase, trainFactor)
    Y_Test_deNorm = getVarDesnormalization(y_Test, meanTarget, stdTarget)
    # for ni in range( 2, maxNumberOfNeuronsByLayer+1):
    for ni in range(2, 9):
        arq = (ni,)
        actFunct = 'tanh'
        for i in range(0, maxNumOfTrials):
            mlpModel = MLPRegressor(solver='adam', hidden_layer_sizes=arq, max_iter=1500, tol=1e-12,
                                    verbose=False,
                                    shuffle=True, activation=actFunct, early_stopping=True, validation_fraction=0.10)
            mlpModel.fit(x_Train, y_Train)
            # currR2Score = mlpModel.score( x_Test, y_Test )
            y_predicted = mlpModel.predict(x_Test)
            y_predicted_deNorm = getVarDesnormalization(y_predicted, meanTarget, stdTarget)
            _, _, r_value, _, _ = stats.linregress(Y_Test_deNorm, y_predicted_deNorm)
            currR2Score = r_value * r_value
            if ((ni == 2) and (i == 0)):
                winR2Score = currR2Score
                winModel = copy.deepcopy(mlpModel)
            else:
                if (currR2Score > winR2Score):
                    winR2Score = currR2Score
                    winModel = copy.deepcopy(mlpModel)
    lstArrImportance2 = []
    for currFeature in range(0, numberOfFeatures):
        x_Test_Temp = copy.deepcopy(x_Test)
        np.random.shuffle(x_Test_Temp[:, currFeature])
        y_predicted = winModel.predict(x_Test_Temp)
        y_predicted_deNorm = getVarDesnormalization(y_predicted, meanTarget, stdTarget)
        _, _, r_value, _, _ = stats.linregress(Y_Test_deNorm, y_predicted_deNorm)
        shuffle_R2 = r_value * r_value
        lstArrImportance2.append((abs(winR2Score - shuffle_R2)) / winR2Score)
    return lstArrImportance2
def main( ROOT_PATH, SIMULATION_NAME1, SIMULATION_NAME2, DATABASENAME,
          numberOfFeatures,featureName,featureByType, colorByFeatureType,
          maxNumOfExp, maxNumOfTrials, trainFactor, method ):

    SIMULATION_PATH_GA  = os.path.join(os.path.sep, ROOT_PATH, SIMULATION_NAME1, SIMULATION_NAME2, 'GAIMPORTANCE')
    SIMULATION_PATH     = os.path.join(os.path.sep, ROOT_PATH, SIMULATION_NAME1)


    databaseFileName = os.path.join(os.path.sep, SIMULATION_PATH, DATABASENAME)
    dfDataBase       = pd.read_csv( databaseFileName )


    allExperiments = range(0,maxNumOfExp)

    pool                   = multiprocessing.Pool( processes=multiprocessing.cpu_count() )
    func                   = partial( f, numberOfFeatures, dfDataBase, maxNumOfTrials,trainFactor)
    AllLstArrImportance2   = pool.map( func, allExperiments )


    arrImportance2 = np.empty((numberOfFeatures, maxNumOfExp))
    for exp in range(0, maxNumOfExp):
        for currFeature in range(0, numberOfFeatures):
            arrImportance2[currFeature, exp] = AllLstArrImportance2[exp][currFeature]

    featureImportanceM2 = np.abs(np.mean(arrImportance2, axis=1))
    minmax = MinMaxScaler()
    featureImportanceM2 = minmax.fit_transform(1 * np.array([featureImportanceM2]).T).T[0]
    featureNameValue = np.array(zip(featureName, featureImportanceM2), dtype=[('Feature', 'S5'), ('Importance', float)])
    featureNameValue[::-1].sort(order='Importance')  # Sort the features by value
    separateFeatureNameValue = zip(*featureNameValue)  # this will be used to plot apply list
    ToolGrafic.saveFeatureImportanceScore(list(separateFeatureNameValue[0]), list(separateFeatureNameValue[1]),
                                          SIMULATION_PATH_GA, method, numberOfFeatures, varColor='coral')

    ToolGrafic.saveFeatureImportanceScoreByType(list(separateFeatureNameValue[0]), list(separateFeatureNameValue[1]),
                                                SIMULATION_PATH_GA, method, len(featureName),
                                                colorByFeatureType, featureByType)

    dfFeaturesScores = pd.concat([pd.DataFrame(list(separateFeatureNameValue[0])),
                                  pd.DataFrame(list(separateFeatureNameValue[1]))], axis=1, ignore_index=True)
    dfFeaturesScores.to_csv(os.path.join(os.path.sep, SIMULATION_PATH_GA, "M2_FeatureImportance.csv"))


    pool.close()
    pool.join()

if __name__ == "__main__":
    args = sys.argv
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
    main( args[1], args[2], args[3], args[4], int(args[5]), featureName,
          featureByType, colorByFeatureType, int(args[6]), int(args[7]),
          float(args[8]), args[9])
