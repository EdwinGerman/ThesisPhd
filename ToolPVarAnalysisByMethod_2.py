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
    lstR2AdjScore      = []
    lstR2Score         = []
    lstR2SpearmanScore = []
    lstListRmse        = []
    for exp in range( 0, maxNumOfExp ):
        x_Train, y_Train, x_Test, y_Test, meanTarget, stdTarget = getScaledDatabase(dfDatabase, trainFactor)
        x_Train = x_Train[:, allFeaturesToTake]
        x_Test = x_Test[:, allFeaturesToTake]
        Y_Test_deNorm =  y_Test * stdTarget + meanTarget  #getVarDesnormalization(y_Test, meanTarget, stdTarget)
        for ni in range(2, maxNumberOfNeuronsByLayer + 1):
            arq = (ni,)
            actFunct = 'tanh'
            for i in range(0, maxNumOfTrials):
                mlpModel = MLPRegressor(solver='adam', hidden_layer_sizes=arq, max_iter=1500, tol=1e-12,
                                        verbose=False,
                                        shuffle=True, activation=actFunct, early_stopping=True,
                                        validation_fraction=0.10)
                mlpModel.fit(x_Train, y_Train)
                y_predicted          = mlpModel.predict(x_Test)
                y_predicted_deNorm   = y_predicted * stdTarget + meanTarget #getVarDesnormalization(y_predicted, meanTarget, stdTarget)
                _, _, r_value, _, _  = stats.linregress(Y_Test_deNorm, y_predicted_deNorm)
                r2_adj = 1 - (1 - r_value * r_value) * (x_Train.shape[0] - 1) / (x_Train.shape[0] - len(allFeaturesToTake) - 1)
                r2 = r_value * r_value
                statistics = stats.spearmanr(Y_Test_deNorm, y_predicted_deNorm)
                r_spearman = statistics[0]
                rmse = rmsCalculation(Y_Test_deNorm, y_predicted_deNorm)
                if ((exp == 0) and (ni == 2) and (i == 0)):
                    winR2AdjScore = r2_adj
                    winR2Score = r2
                    winRSpearman = r_spearman
                    winRmse = rmse
                else:
                    if (r2_adj > winR2AdjScore):
                        winR2Score = r2_adj
                    if (r2 > winR2Score):
                        winR2Score = r2
                    if (r_spearman > winRSpearman):
                        winRSpearman = r_spearman
                    if (rmse < winRmse):
                        winRmse = rmse
                del mlpModel
        lstR2AdjScore.append(winR2AdjScore)
        lstR2Score.append(winR2Score)
        lstR2SpearmanScore.append(winRSpearman)
        lstListRmse.append(winRmse)
    nGroup       = len( allFeaturesToTake )
    meanAdjR2    = np.mean( lstR2AdjScore )
    meanR2       = np.mean( lstR2Score )
    meanSpearman = np.mean( lstR2SpearmanScore )
    meanRmse     = np.mean( lstListRmse )
    ####End Process
    outPutList = [nGroup, meanAdjR2,meanR2, meanSpearman, meanRmse ]
    return outPutList

def main( ROOT_PATH, SIMULATION_NAME1, SIMULATION_NAME2, DATABASENAME, strMethodFeatureRanking, strFeaturesName,
          numberOfFeatures, maxNumberOfNeuronsByLayer, maxNumOfExp, maxNumOfTrials, trainFactor, method ):

    SIMULATION_PATH_GA  = os.path.join(os.path.sep, ROOT_PATH, SIMULATION_NAME1, SIMULATION_NAME2, 'GAIMPORTANCE')
    SIMULATION_PATH     = os.path.join(os.path.sep, ROOT_PATH, SIMULATION_NAME1)

    methodFeatureRanking = strMethodFeatureRanking.split(",")

    databaseFileName = os.path.join(os.path.sep, SIMULATION_PATH, DATABASENAME)
    dfDataBase       = pd.read_csv( databaseFileName )

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
    varListR2AdjValue      = list( tempDf.ix[:,1] )
    varListR2Value         = list( tempDf.ix[:,2] )
    varListR2SpearmanValue = list( tempDf.ix[:,3] )
    varListRmse            = list( tempDf.ix[:,4] )


    ToolGrafic.saveAnalysisByMethod( varList, varListR2Value, "R2", method, SIMULATION_PATH_GA)
    ToolGrafic.saveAnalysisByMethod( varList, varListR2AdjValue, "RA", method, SIMULATION_PATH_GA)
    ToolGrafic.saveAnalysisByMethod( varList, varListR2SpearmanValue, "RE", method, SIMULATION_PATH_GA)
    ToolGrafic.saveAnalysisByMethod( varList, varListRmse, "RMSE", method, SIMULATION_PATH_GA)

    ToolGrafic.saveAnalysisByMethod2( varList, varListR2AdjValue, varListR2Value, method, SIMULATION_PATH_GA )

    df = pd.concat([pd.DataFrame(varList),pd.DataFrame(varListR2AdjValue), pd.DataFrame(varListR2Value),
                    pd.DataFrame(varListR2SpearmanValue), pd.DataFrame(varListRmse)], ignore_index=True, axis=1)
    df.to_csv( os.path.join( os.path.sep, SIMULATION_PATH_GA, method +"_EPS_NumVarAnalysis.csv" ) )


    pool.close()
    pool.join()

if __name__ == "__main__":
    args = sys.argv
    main( args[1], args[2], args[3], args[4], args[5], args[6], int(args[7]), int(args[8]), int(args[9]), int(args[10]), float(args[11]), args[12] )