import multiprocessing
import numpy as np
import os
import pandas as pd
import sys
from functools import partial
from scipy import stats
import scipy
import sklearn
from sklearn import preprocessing
from sklearn.neural_network import MLPRegressor

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

def getScaledDataBase2 ( dfDataBase, trainFactor):
    dfProProteina = sklearn.utils.shuffle(dfDataBase)
    dfProProteina = dfProProteina.reset_index(drop=True)
    arrayDatabase = dfProProteina.as_matrix()
    arrayDatabase = arrayDatabase[1:arrayDatabase.shape[0], :]

    rowsTestSet         = int(arrayDatabase.shape[0] * trainFactor)
    rowsTrainSet        = arrayDatabase.shape[0] - rowsTestSet

    dfTrainSet          = arrayDatabase[0:rowsTrainSet, :]
    dfTestSet           = arrayDatabase[rowsTrainSet:arrayDatabase.size[0], :]



    scaler              = preprocessing.StandardScaler().fit(dfTrainSet)

    scaled_database     = scaler.transform(dfTrainSet)

    x_Train             = scaled_database[:, 0:scaled_database.shape[1] - 1]
    y_Train             = scaled_database[:, scaled_database.shape[1] - 1]

    meanTarget          = scaler.mean_[scaled_database.shape[1] - 1]
    stdTarget           = scaler.scale_[scaled_database.shape[1] - 1]

    x_Test              = dfTestSet[:, 0:scaled_database.shape[1] - 1]
    y_Test              = dfTestSet[:, scaled_database.shape[1] - 1]

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
def f( maxNumberOfNeuronsByLayer, dfDatabase,
       maxNumOfExp,maxNumOfTrials,trainFactor,
       allFeaturesToTake ):
    ####Begin Process
    scipy.random.seed()
    lstError = []
    for exp in range( 0, maxNumOfExp ):
        #x_Train, y_Train, x_Test, y_Test, meanTarget, stdTarget = getScaledDatabase(dfDatabase, trainFactor)
        x_Train        =  x_Train[:, allFeaturesToTake]
        x_Test         =  x_Test[:, allFeaturesToTake]
        Y_Test_deNorm  =  y_Test * stdTarget + meanTarget # getVarDesnormalization(y_Test, meanTarget, stdTarget)
        for ni in range(2, maxNumberOfNeuronsByLayer + 1):
            arq = (ni,)
            actFunct = 'tanh'
            for i in range(0, maxNumOfTrials):
                mlpModel = MLPRegressor(solver='adam', hidden_layer_sizes=arq, max_iter=1500, tol=1e-12,
                                        verbose=False,
                                        shuffle=True, activation=actFunct, early_stopping=True,
                                        validation_fraction=0.10)
                mlpModel.fit(x_Train, y_Train)
                y_predicted        =  mlpModel.predict(x_Test)
                y_predicted_deNorm =  y_predicted * stdTarget + meanTarget #getVarDesnormalization(y_predicted, meanTarget, stdTarget)
                lstError1          =  list((Y_Test_deNorm - y_predicted_deNorm) * (Y_Test_deNorm - y_predicted_deNorm))
                currError          =  np.mean(lstError1)

                if ((exp == 0) and (ni == 2) and (i == 0)):
                    winError = currError
                else:
                    if currError < winError:
                        winError = currError
                del mlpModel
        lstError.append( winError )
    nGroup       = len( allFeaturesToTake )
    meanError    = np.mean( lstError )
    ####End Process
    outPutList = [nGroup, meanError ]
    return outPutList

def main( ROOT_PATH, SIMULATION_NAME1, SIMULATION_NAME2, DATABASENAME, strMethodFeatureRanking, strFeaturesName,
          numberOfFeatures, maxNumberOfNeuronsByLayer, maxNumOfExp, maxNumOfTrials, trainFactor, method ):

    SIMULATION_PATH_GA  = os.path.join(os.path.sep, ROOT_PATH, SIMULATION_NAME1, SIMULATION_NAME2, 'GAIMPORTANCE')
    SIMULATION_PATH     = os.path.join(os.path.sep, ROOT_PATH, SIMULATION_NAME1)

    methodFeatureRanking = strMethodFeatureRanking.split(",")

    databaseFileName = os.path.join(os.path.sep, SIMULATION_PATH, DATABASENAME)
    dfDataBase       = pd.read_csv( databaseFileName )




    x_Train, y_Train, x_Test, y_Test, meanTarget, stdTarget = getScaledDatabase(dfDataBase, trainFactor)

    featuresName = strFeaturesName.split(",")
    allFeaturesToTake = []
    for selFeature  in range( 0, numberOfFeatures-1):
        featureToTake =  methodFeatureRanking[0:selFeature+2]
        lstIdxFeature = []
        for currFeature in featureToTake:
            idx = 0
            for feature in featuresName:
                if ( currFeature==feature ):
                    lstIdxFeature.append( idx )
                    break
                idx = idx + 1
        allFeaturesToTake.append( lstIdxFeature )


    pool                   = multiprocessing.Pool( processes=multiprocessing.cpu_count() )
    func                   = partial( f, maxNumberOfNeuronsByLayer, dfDataBase, maxNumOfExp, maxNumOfTrials, trainFactor )
    lstSensivityAnalysis   = pool.map( func, allFeaturesToTake )


    tempDf                 = pd.DataFrame()
    for lstAnalysis in lstSensivityAnalysis:
        tempDf = pd.concat([tempDf, pd.DataFrame(lstAnalysis)], axis=1, ignore_index=True)
    tempDf = tempDf.T
    varList                = list( tempDf.ix[:,0] )
    varError               = list( tempDf.ix[:,1] )

    ToolGrafic.saveAnalysisByMethod(varList, varError, "ERROR", method, SIMULATION_PATH_GA)
    df = pd.DataFrame(varError)
    df.to_csv(os.path.join(os.path.sep, SIMULATION_PATH_GA, method + "_EPS_NumVarAnalysis.csv"))

    pool.close()
    pool.join()

if __name__ == "__main__":
    args = sys.argv
    main( args[1], args[2], args[3], args[4], args[5], args[6], int(args[7]), int(args[8]), int(args[9]), int(args[10]), float(args[11]), args[12] )