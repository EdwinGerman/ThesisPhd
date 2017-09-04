# Using the magic encoding
# -*- coding: utf-8 -*-
import copy
import csv
import os
import pickle
import random
import shutil
import time

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import minepy
import numpy as np
import pandas as pd
import sklearn.utils
from deap import creator, base, tools, algorithms
#from deap import creator, base
from scipy import stats
from scoop import futures
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE, f_regression
from sklearn.linear_model import (LinearRegression, Ridge,
                                  Lasso, RandomizedLasso)
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
import multiprocessing
from timeit import default_timer as timer
import ToolGrafic


#creator = None
def checkActiveFeatures(minNumOfOnes):
    def decorator(func):
        def wrapper(*args, **kargs):
            offspring = func(*args, **kargs)
            for child in offspring:
                len_child = len(child)
                countOne  = 0
                for i in xrange(len_child):
                    if child[i]==1:
                        countOne = countOne + 1
                        if countOne>=minNumOfOnes:
                            break
                if countOne<2:
                    if countOne==1: # only 1 gene must be activated
                        values =  np.array(child)
                        ii     = int(np.where(values == 1)[0])
                        child[ii]=0
                    midPoint = int(len_child/2.0)
                    pos1 = random.randint(0, midPoint)
                    pos2 = random.randint(midPoint+1, len_child-1)
                    child[pos1] = 1
                    child[pos2] = 1
            return offspring
        return wrapper
    return decorator
def checkActiveFeatures3(minNumOfOnes):
    def decorator(func):
        def wrapper(*args, **kargs):
            pop = func(*args, **kargs)
            for ind in pop:
                positionOfOnes = [i for i, j in enumerate(ind) if j == 1]
                if (len(positionOfOnes))>3:
                   numForDesactivation = len(positionOfOnes)-3
                   idxToDesactivate = random.sample(range(0, len(positionOfOnes)), numForDesactivation)
                   for idx in idxToDesactivate:
                       ind[positionOfOnes[idx]]=0
            return pop
        return wrapper
    return decorator
creator.create("Fitness", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.Fitness)
def set_creator(cr):
    global creator
    creator = cr
set_creator(creator)
numberOfFeatures=37
toolbox = base.Toolbox()

toolbox.register("attr_bool", random.randint, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=numberOfFeatures)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mate", tools.cxOrdered)
# toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("mutate", tools.mutUniformInt, low=0, up=1, indpb=0.1)

toolbox.decorate("population", checkActiveFeatures3(2))

toolbox.decorate("mate", checkActiveFeatures(2))
toolbox.decorate("mutate", checkActiveFeatures(2))
toolbox.register("select", tools.selTournament, tournsize=2)
toolbox.register("map", futures.map)

def rank_to_dict( ranks, names, order=1 ):
    minmax = MinMaxScaler()
    ranks = minmax.fit_transform(order * np.array([ranks]).T).T[0]
    ranks = map(lambda x: round(x, 4), ranks)
    return dict(zip(names, ranks))

def checkActiveFeatures2(minNumOfOnes):
    def decorator(func):
        def wrapper(*args, **kargs):
            pop = func(*args, **kargs)
            for ind in pop:
                len_ind = len(ind)
                countOne = 0
                for i in xrange(len_ind):
                    if ind[i] == 1:
                        countOne = countOne + 1
                        if countOne >= minNumOfOnes:
                            break
                if countOne<2:
                    if countOne==1: # only 1 gene must be activated
                        values =  np.array(ind)
                        ii     =  np.where(values == 1)[0]
                        ind[ii]=  0
                    midPoint = int(len_ind/2.0)
                    pos1 = random.randint(0, midPoint)
                    pos2 = random.randint(midPoint+1, len_ind-1)
                    ind[pos1] = 1
                    ind[pos2] = 1
            return pop
        return wrapper
    return decorator

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
def NNEvolutionPlot( lstNNEvolution, predicted_values, target_values, r_square_winner, slope_winner,
                     intercept_winner, r_adjusted_winner, r_spearman_winner, rho_spearman_winner,
                     rmse_winner,exp, gen, savePath ):
    plt.plot(lstNNEvolution, color='red', linewidth=2)
    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    plt.title('Neural Network Evolution')
    plt.savefig(os.path.join(os.path.sep, savePath, "EPS_NNEvolution_en_"+str(exp)+"_"+str(gen)+".eps"), dpi=1200)
    plt.xlabel(u'Épocas')
    plt.ylabel(u'MSE')
    plt.title(u'Evolução da Rede Neural')
    plt.savefig(os.path.join(os.path.sep, savePath, "EPS_NNEvolution_pt_"+str(exp)+"_"+str(gen)+".eps"), dpi=1200)
    plt.cla()
    plt.close()
    plt.plot( predicted_values , color='blue')
    plt.plot( target_values, color='green')
    minValue = 0.0
    if np.min(predicted_values)<np.min(target_values):
        minValue = np.min(predicted_values)
    else:
        minValue = np.min(target_values)
    maxValue = 0.0
    if np.max(predicted_values)<np.max(target_values):
        maxValue = np.max(predicted_values)
    else:
        maxValue = np.max(target_values)
    heightValue = (minValue + maxValue)/2.0
    error2 = (target_values-predicted_values)*(target_values-predicted_values)
    plt.plot(heightValue+error2, color='red')
    plt.legend(('Predicted Values', 'Target Values', "Square Error - "+('$RMSE$={}'.format(rmse_winner))), loc="best", prop={'size':10})
    plt.xlabel('Examples')
    plt.ylabel('GTS-SCORE')
    plt.title('Neural Network Fit Analysis in Test Set')
    plt.savefig(os.path.join(os.path.sep, savePath, "EPS_NNFitAnalysis_en_"+str(exp)+"_"+str(gen)+".eps"), dpi=1200)
    plt.xlabel(u'Exemplos')
    plt.ylabel(u'GTS-SCORE')
    plt.title(u'Análise de ajuste de rede neural no conjunto de teste')
    plt.savefig(os.path.join(os.path.sep, savePath, "EPS_NNFitAnalysis_pt_"+str(exp)+"_"+str(gen)+".eps"), dpi=1200 )
    plt.cla()
    plt.close()
    plt.scatter( target_values, predicted_values )
    plt.xlabel('Target Values')
    plt.ylabel('Predicted Values')
    plt.title('Target Values vs Predicted Values')
    plt.savefig(os.path.join(os.path.sep, savePath, "EPS_NNPredictedVsTarget_en_"+str(exp)+"_"+str(gen)+".eps"), dpi=1200 )
    plt.cla()
    plt.close()
    plt.scatter( target_values, predicted_values, color="red")
    yFit = [ intercept_winner + slope_winner *xi for xi in target_values]
    plt.plot(target_values, yFit, color="green", label='R2')
    xCoord  = np.min(target_values)
    yCoord  = np.max(predicted_values)
    plt.text(xCoord, yCoord,'$R^2$   ={}'.format(round(r_square_winner,10)), verticalalignment='top', fontsize=10)
    plt.text(xCoord, yCoord-0.05, '$R_a^2$   ={}'.format(round(r_adjusted_winner, 10)), verticalalignment='top', fontsize=10)
    #plt.text(xCoord, yCoord-0.280, '$R_s$    ={}'.format(round(r_spearman_winner, 10)), verticalalignment='top', fontsize=10)
    #plt.text(xCoord, yCoord-0.415, '$Rho$  ={}'.format(rho_spearman_winner), verticalalignment='top', fontsize=10)
    #plt.text(xCoord, yCoord-0.560, '$RMSE$  ={}'.format(rmse_winner, 10), verticalalignment='top', fontsize=10)
    plt.xlabel('Target Values')
    plt.ylabel('Predicted Values')
    plt.title('Scatter Analysis')
    plt.savefig(os.path.join(os.path.sep, savePath, "EPS_NNScatter_en"+str(exp)+"_"+str(gen)+".eps"), dpi=1200)
    plt.cla()
    plt.close()
    columnsHeader = ['PREDICTED','TARGET']
    dfDataPredictedVsReal = pd.concat([pd.DataFrame(predicted_values),pd.DataFrame(target_values)], axis=1, ignore_index=True)
    dfDataPredictedVsReal.to_csv(os.path.join(os.path.sep, savePath, "EPS_NNPredictedVsTargetData" + str(exp) + "_" + str(gen) + ".csv"), header=columnsHeader)

def finalNNTrain( x_Train, y_Train, x_Test, y_Test,
                  minNumberOfNeuronsByLayer, maxNumberOfNeuronsByLayer, idxSelectedFeat, SIMULATION_PATH_NN):
    X_train = x_Train[:, idxSelectedFeat]
    Y_train = y_Train
    X_Test = x_Test[:, idxSelectedFeat]
    Y_Test = y_Test
    n = X_Test.shape[0]
    for nnExp in range( 0,30 ):
        geneArq_1 = random.randint(minNumberOfNeuronsByLayer, maxNumberOfNeuronsByLayer)
        geneArq_2 = random.randint(minNumberOfNeuronsByLayer, maxNumberOfNeuronsByLayer)
        geneArq_3 = random.randint(31, 33)
        if (geneArq_3 == 31):
            actFunct = 'logistic'
        elif (geneArq_3 == 32):
            actFunct = 'tanh'
        elif (geneArq_3 == 33):
            actFunct = 'relu'
        mlpModel = MLPRegressor(solver='adam', hidden_layer_sizes=(geneArq_1, geneArq_2),
                                max_iter=1500, tol=1e-12, verbose=True, shuffle=True,
                                activation=actFunct, early_stopping=True, validation_fraction=0.10)
        mlpModel.fit(X_train, Y_train)
        y_predicted = mlpModel.predict(X_Test)
        slope, intercept, r_value, p_value, std_err = stats.linregress(Y_Test, y_predicted)
        r_square = r_value * r_value
        r_adjusted = 1 - ((1 - r_square) * (n - 1) / (n - numberOfFeatures - 1))
        statistics = stats.spearmanr(Y_Test, y_predicted)
        r_spearman = statistics[0]
        rho_spearman = statistics[1]
        lstLossCurve = mlpModel.loss_curve_
        rmse = rmsCalculation(Y_Test, y_predicted)
        if ( nnExp==0 ):
            mlpModelWinner = copy.deepcopy(mlpModel)
            y_predictedWinner  = y_predicted
            slopeWinner        = slope
            interceptWinner    = intercept
            r_valueWinner      = r_value
            p_valueWinner      = p_value
            std_errWinner      = std_err
            r_squareWinner     = r_square
            r_adjustedWinner   = r_adjusted
            r_spearmanWinner   = r_spearman
            rho_spearmanWinner = rho_spearman
            lstLossCurveWinner = lstLossCurve
            rmseWinner         = rmse
            actFunctWinner     = actFunct
            numberNeurons1     =geneArq_1
            numberNeurons2     = geneArq_2
        else:
            if ( rmse < rmseWinner ):
                mlpModelWinner = copy.deepcopy(mlpModel)
                y_predictedWinner  = y_predicted
                slopeWinner        = slope
                interceptWinner    = intercept
                r_valueWinner      = r_value
                p_valueWinner      = p_value
                std_errWinner      = std_err
                r_squareWinner     = r_square
                r_adjustedWinner   = r_adjusted
                r_spearmanWinner   = r_spearman
                rho_spearmanWinner = rho_spearman
                lstLossCurveWinner = lstLossCurve
                rmseWinner         = rmse
                actFunctWinner = actFunct
                numberNeurons1 = geneArq_1
                numberNeurons2 = geneArq_2
    NNEvolutionPlot( lstLossCurveWinner, y_predictedWinner, Y_Test, r_squareWinner, slopeWinner, interceptWinner,
                     r_adjustedWinner, rho_spearmanWinner, rho_spearmanWinner, rmseWinner,99,99, SIMULATION_PATH_NN )
    strNNTitle = 'Slope' + ',' + 'Intercep' + ',' + 'r_value' + ',' + 'p_value' + ',' + 'std_err' + ',' + \
                 'r_square' + ',' + 'r_adjusted' + ',' + 'r_spearman' + ',' + 'rho_Spearman' + ',' + \
                 'ActFunt'+','+'NeuNum1'+','+'NeuNum2'+','+'RMSE' +'\n'
    f = open(os.path.join(SIMULATION_PATH_NN, "summaryBestNN.csv"), "a")
    f.write(strNNTitle)
    f.close()
    f1 = open(os.path.join(SIMULATION_PATH_NN, "summaryBestNN.csv"), "a")
    strData =str(slopeWinner)+','+str(interceptWinner)+','+str(r_valueWinner)+','+str(p_valueWinner)+','+str(std_errWinner)+ \
             ','+str(r_squareWinner)+','+str(r_adjustedWinner)+','+str(r_spearmanWinner)+','+\
             str(rho_spearmanWinner)+','+actFunctWinner+','+str(numberNeurons1)+','+str(numberNeurons2)+','+str(rmseWinner)+'\n'
    f1.write( strData )
    f1.close()
    modelName = 'Best_NNMODEL.sav'
    pickle.dump( mlpModelWinner, open( os.path.join(SIMULATION_PATH_NN, modelName ), 'wb' ) )
def getVarDesnormalization( arrVar, varMean, varStd):
    retunedValue = arrVar*varStd+varMean
    return retunedValue
def getFitness( x_Train, y_Train, x_Test, y_Test, meanTarget, stdTarget,
                minNumberOfNeuronsByLayer, maxNumberOfNeuronsByLayer,
                individual, idxSelectedFeat=[], boolSelected=False ):
    if ( boolSelected==True ):
        idxSelectedFeatures = idxSelectedFeat
    else:
        idxSelectedFeatures = [i for i, j in enumerate( individual ) if j == 1]
    X_train = x_Train[:, idxSelectedFeatures]
    Y_train = y_Train
    X_Test  = x_Test[:, idxSelectedFeatures]
    Y_Test  = y_Test
    n = X_Test.shape[0]
    geneArq_1 = random.randint( minNumberOfNeuronsByLayer, maxNumberOfNeuronsByLayer )
    geneArq_2 = random.randint( minNumberOfNeuronsByLayer, maxNumberOfNeuronsByLayer )
    geneArq_3 = random.randint( 31, 33 )
    individual.geneArq_1 = geneArq_1
    individual.geneArq_2 = geneArq_2
    individual.geneArq_3 = geneArq_3
    if ( geneArq_3==31 ):
        actFunct ='tanh'
    elif( geneArq_3==32 ):
        actFunct ='tanh'
    elif( geneArq_3==33 ):
        actFunct ='tanh'
    mlpModel = MLPRegressor( solver='adam', hidden_layer_sizes=( geneArq_1, geneArq_2 ), max_iter=1000, tol=1e-8, verbose=False,
                             shuffle=True, activation=actFunct, early_stopping=True, validation_fraction=0.10 )
    mlpModel.fit( X_train, Y_train )
    y_predicted        = mlpModel.predict( X_Test )

    y_predicted_deNorm = getVarDesnormalization(y_predicted,meanTarget,stdTarget)
    Y_Test_deNorm      = getVarDesnormalization(Y_Test,meanTarget,stdTarget)

    slope, intercept, r_value, p_value, std_err = stats.linregress( Y_Test_deNorm , y_predicted_deNorm )
    r_square     = r_value * r_value
    r_adjusted   = 1 - ( ( 1 - r_square ) * ( n - 1 ) / ( n - numberOfFeatures - 1 ) )
    statistics   = stats.spearmanr( Y_Test_deNorm, y_predicted_deNorm )
    r_spearman   = statistics[0]
    rho_spearman = statistics[1]
    lstLossCurve = mlpModel.loss_curve_
    rmse         = rmsCalculation( Y_Test_deNorm, y_predicted_deNorm )
    arrStatistics     = np.empty((1, 10 ))
    arrStatistics[0,0]= slope
    arrStatistics[0,1]= intercept
    arrStatistics[0,2]= r_value
    arrStatistics[0,3]= p_value
    arrStatistics[0,4]= std_err
    arrStatistics[0,5]= r_square
    arrStatistics[0,6]= r_adjusted
    arrStatistics[0,7]= r_spearman
    arrStatistics[0,8]= rho_spearman
    arrStatistics[0,9]= rmse
    individual.ArrStats     = arrStatistics
    individual.lstLostCurve = copy.deepcopy(lstLossCurve)
    individual.nnModel      = copy.deepcopy(mlpModel)
    individual.Y_predicted  = copy.deepcopy(list(y_predicted_deNorm))
    individual.Y_test       = copy.deepcopy(list(Y_Test_deNorm))
    return rmse,
def getFitness2( x_Train, y_Train, x_Test, y_Test, meanTarget, stdTarget,
                 minNumberOfNeuronsByLayer, maxNumberOfNeuronsByLayer,
                 individual, idxSelectedFeat=[], boolSelected=False ):
    if ( boolSelected==True ):
        idxSelectedFeatures = idxSelectedFeat
    else:
        idxSelectedFeatures = [i for i, j in enumerate( individual ) if j == 1]
    X_train = x_Train[:, idxSelectedFeatures]
    Y_train = y_Train
    X_Test  = x_Test[:, idxSelectedFeatures]
    Y_Test  = y_Test
    n = X_Test.shape[0]
    #geneArq_1 = random.randint( minNumberOfNeuronsByLayer, maxNumberOfNeuronsByLayer )
    #geneArq_2 = random.randint( minNumberOfNeuronsByLayer, maxNumberOfNeuronsByLayer )
    geneArq_1 = 0
    geneArq_2 = 0
    geneArq_3 = 31
    individual.fitness.AddGene3 = geneArq_3
    if ( geneArq_3==31 ):
        actFunct ='tanh'
    Y_Test_deNorm = getVarDesnormalization( Y_Test, meanTarget, stdTarget )
    chance = random.random()
    if ( chance < 0.0000000001 ):
        print("Complex Fitness!!!")
        for ni in range( 0, maxNumberOfNeuronsByLayer+1 ):
            for nj in range( 2, maxNumberOfNeuronsByLayer+1 ):
                if ni == 0: #only 1 layer
                    arq       = ( nj,)
                    #arq       = (numberOfFeatures, nj, 1)
                else:  #two layers
                    arq       = ( ni, nj )
                    #arq       = ( numberOfFeatures, ni, nj, 1)
                for attempt in range( 0, 10 ):
                    mlpModel = MLPRegressor( solver='adam', hidden_layer_sizes=arq, max_iter=1500,
                                             tol=1e-8, verbose=False,
                                             shuffle=True, batch_size=200, activation=actFunct, early_stopping=True,
                                             validation_fraction=0.10 )
                    mlpModel.fit( X_train, Y_train )
                    y_predicted        = mlpModel.predict(X_Test)
                    y_predicted_deNorm = getVarDesnormalization(y_predicted, meanTarget, stdTarget)
                    rmse               = rmsCalculation(Y_Test_deNorm, y_predicted_deNorm)
                    if ( (ni==0) and (nj==2)):
                        rmseWinner               = rmse
                        modelWinner              = copy.deepcopy(mlpModel)
                        y_predicted_deNormWinner = y_predicted_deNorm
                        geneArq_1                = ni
                        geneArq_2                = nj
                    else:
                        if (rmse<rmseWinner):
                            rmseWinner               = rmse
                            modelWinner              = copy.deepcopy(mlpModel)
                            y_predicted_deNormWinner = y_predicted_deNorm
                            geneArq_1                = ni
                            geneArq_2                = nj
        print("End Complex Fitness!!!")
    else:
        ni = random.randint( 0, maxNumberOfNeuronsByLayer )
        nj = random.randint( 2, maxNumberOfNeuronsByLayer )
        if ni == 0:  # only 1 layer
            arq = (nj,)
            # arq       = (numberOfFeatures, nj, 1)
        else:  # two layers
            arq = (ni, nj)
        for attempt in range(0, 5):
            mlpModel = MLPRegressor(solver='adam', hidden_layer_sizes=arq, max_iter=1500,
                                    tol=1e-8, verbose=False,
                                    shuffle=True, batch_size=200, activation=actFunct, early_stopping=True,
                                    validation_fraction=0.10)
            mlpModel.fit(X_train, Y_train)
            y_predicted = mlpModel.predict(X_Test)
            y_predicted_deNorm = getVarDesnormalization(y_predicted, meanTarget, stdTarget)
            rmse = rmsCalculation(Y_Test_deNorm, y_predicted_deNorm)
            if (attempt ==0):
                rmseWinner = rmse
                modelWinner = copy.deepcopy(mlpModel)
                y_predicted_deNormWinner = y_predicted_deNorm
                geneArq_1 = ni
                geneArq_2 = nj
            else:
                if (rmse < rmseWinner):
                    rmseWinner = rmse
                    modelWinner = copy.deepcopy(mlpModel)
                    y_predicted_deNormWinner = y_predicted_deNorm
                    geneArq_1 = ni
                    geneArq_2 = nj
    individual.fitness.AddGene1 = geneArq_1
    individual.fitness.AddGene2 = geneArq_2
    slope, intercept, r_value, p_value, std_err = stats.linregress( Y_Test_deNorm , y_predicted_deNormWinner )
    r_square     = r_value * r_value
    r_adjusted   = 1 - ( ( 1 - r_square ) * ( n - 1 ) / ( n - numberOfFeatures - 1 ) )
    statistics   = stats.spearmanr( Y_Test_deNorm, y_predicted_deNormWinner )
    r_spearman   = statistics[0]
    rho_spearman = statistics[1]
    lstLossCurve = modelWinner.loss_curve_
    arrStatistics     = np.empty((1, 10 ))
    arrStatistics[0,0]= slope
    arrStatistics[0,1]= intercept
    arrStatistics[0,2]= r_value
    arrStatistics[0,3]= p_value
    arrStatistics[0,4]= std_err
    arrStatistics[0,5]= r_square
    arrStatistics[0,6]= r_adjusted
    arrStatistics[0,7]= r_spearman
    arrStatistics[0,8]= rho_spearman
    arrStatistics[0,9]= rmseWinner

    individual.fitness.ArrStats     = arrStatistics
    individual.fitness.lstLostCurve = copy.deepcopy(lstLossCurve)
    individual.fitness.nnModel      = copy.deepcopy(modelWinner)
    individual.fitness.Y_predicted  = copy.deepcopy(list(y_predicted_deNormWinner))
    individual.fitness.Y_test       = copy.deepcopy(list(Y_Test_deNorm))
    return rmseWinner, geneArq_1, geneArq_2, geneArq_3,arrStatistics,copy.deepcopy(lstLossCurve),copy.deepcopy(modelWinner),copy.deepcopy(list(y_predicted_deNormWinner)),copy.deepcopy(list(Y_Test_deNorm)),
def getNewIndividualsNumber( numberGeneration, currGeneration, minPercentage, maxPercentage, popSize):
    minInd = int(minPercentage * popSize)
    maxInd = int(maxPercentage * popSize)
    numberNewIndividuals = ((maxInd-minInd)/numberGeneration)*currGeneration+ minInd
    return numberNewIndividuals
def getIndWithArchitecture (individual ):
    indArr   = np.array( individual )
    gene1    = individual.fitness.AddGene1 #TODO: Add fitness
    gene2    = individual.fitness.AddGene2
    gene3    = individual.fitness.AddGene3
    indArr2  = np.array([gene1, gene2, gene3])
    indFinal = np.hstack((indArr, indArr2))
    return  indFinal

def getImportanceMethod2( dfDatabase,  numberOfFeatures, featureName, minNumberOfNeuronsByLayer, maxNumberOfNeuronsByLayer,
                           maxNumOfExp, maxNumOfTrials, trainFactor, colorByFeatureType, featureByType, SIMULATION_PATH_GA, individual=[]):
    #actFunct     = ""
    arrImportance2 = np.empty((numberOfFeatures, maxNumOfExp))
    for exp in range(0, maxNumOfExp):
        x_Train, y_Train, x_Test, y_Test, meanTarget, stdTarget = getScaledDatabase(dfDatabase, trainFactor)
        Y_Test_deNorm = getVarDesnormalization(y_Test, meanTarget, stdTarget)
        #for ni in range( 2, maxNumberOfNeuronsByLayer+1):
        for ni in range(2, 9):
            arq      = ( ni,  )
            actFunct = 'tanh'
            for i in range(0, maxNumOfTrials):
                mlpModel = MLPRegressor( solver='adam', hidden_layer_sizes=arq, max_iter=1500, tol=1e-12,
                                         verbose=False,
                                         shuffle=True, activation=actFunct, early_stopping=True, validation_fraction=0.10)
                mlpModel.fit(x_Train, y_Train)
                #currR2Score = mlpModel.score( x_Test, y_Test )
                y_predicted         = mlpModel.predict(x_Test)
                y_predicted_deNorm  = getVarDesnormalization(y_predicted, meanTarget, stdTarget)
                _, _, r_value, _, _ = stats.linregress(Y_Test_deNorm, y_predicted_deNorm)
                currR2Score   = r_value*r_value
                if ( ( ni ==2 ) and ( i ==0 ) ):
                    winR2Score = currR2Score
                    winModel   = copy.deepcopy(mlpModel)
                else:
                    if ( currR2Score>winR2Score ):
                        winR2Score = currR2Score
                        winModel   = copy.deepcopy(mlpModel)
        for currFeature in range( 0, numberOfFeatures ):
            x_Test_Temp                    = copy.deepcopy( x_Test )
            np.random.shuffle(x_Test_Temp[:, currFeature])
            y_predicted                    = winModel.predict( x_Test_Temp )
            y_predicted_deNorm             = getVarDesnormalization(y_predicted, meanTarget, stdTarget)
            _, _, r_value, _, _            = stats.linregress(Y_Test_deNorm, y_predicted_deNorm)
            shuffle_R2                     = r_value * r_value
            arrImportance2[currFeature,exp]= ( abs(winR2Score-shuffle_R2) ) /winR2Score
        print('Method 2...'+'  '+ '{0:4} ==> {1:4d}'.format("Exp Number", exp) )

    featureImportanceM2      = np.abs(np.mean(arrImportance2, axis=1))
    minmax                   = MinMaxScaler()
    featureImportanceM2      = minmax.fit_transform(1 * np.array([featureImportanceM2]).T).T[0]
    featureNameValue         = np.array(zip(featureName, featureImportanceM2), dtype=[('Feature', 'S5'), ('Importance', float)])
    featureNameValue[::-1].sort(order='Importance')  # Sort the features by value
    separateFeatureNameValue = zip(*featureNameValue)  # this will be used to plot apply list
    ToolGrafic.saveFeatureImportanceScore(list(separateFeatureNameValue[0]), list(separateFeatureNameValue[1]),
                                          SIMULATION_PATH_GA, 'M2', numberOfFeatures, varColor='coral')

    ToolGrafic.saveFeatureImportanceScoreByType(list(separateFeatureNameValue[0]), list(separateFeatureNameValue[1]),
                                                SIMULATION_PATH_GA, 'M2', len(featureName),
                                                colorByFeatureType, featureByType)

    dfFeaturesScores = pd.concat([pd.DataFrame(list(separateFeatureNameValue[0])),
                                  pd.DataFrame(list(separateFeatureNameValue[1]))], axis=1, ignore_index=True)
    dfFeaturesScores.to_csv(os.path.join(os.path.sep, SIMULATION_PATH_GA, "M2_FeatureImportance.csv"))


    varAnalysisByMethod_2( dfDatabase, list(separateFeatureNameValue[0]), featureName, maxNumberOfNeuronsByLayer=8,
                                maxNumOfExp=1, maxNumOfTrials=1, trainFactor=trainFactor,
                                method="M2", SIMULATION_PATH_GA=SIMULATION_PATH_GA)

    # for exp in range( 0,maxNumOfExp  ):
    #     x_Train, y_Train, x_Test, y_Test, meanTarget, stdTarget = getScaledDatabase( dfDatabase, trainFactor )
    #     avgTrainR2Score = 0.0
    #     for i in range( 0,maxNumOfTrials ):
    #         geneArq_1 = 0
    #         geneArq_2 = 0
    #         geneArq_3 = 31
    #         if ( geneArq_3 == 31 ):
    #             actFunct = 'tanh'
    #         mlpModel = MLPRegressor( solver='adam', hidden_layer_sizes=(geneArq_1, geneArq_2), max_iter=1500, tol=1e-12,
    #                                  verbose=False,
    #                                  shuffle=True, activation=actFunct, early_stopping=True, validation_fraction=0.10 )
    #         mlpModel.fit( x_Train, y_Train )
    #         currR2Score = mlpModel.score( x_Test, y_Test )
    #         if ( i == 0 ):
    #             winR2Score       =  currR2Score
    #             winModel         =  copy.deepcopy( mlpModel )
    #         else:
    #             if ( currR2Score>winR2Score ):
    #                 winR2Score = currR2Score
    #                 winModel   = copy.deepcopy( mlpModel )
    #         avgTrainR2Score    =  avgTrainR2Score  +  currR2Score
    #     avgTrainR2Score   = avgTrainR2Score/maxNumOfTrials
    #     if exp==0:
    #         arrImportance2    = np.empty( ( numberOfFeatures, maxNumOfExp ) )
    #     for currFeature in range( 0, numberOfFeatures ):
    #         x_Test_Temp                    = copy.deepcopy( x_Test )
    #         np.random.shuffle(x_Test_Temp[:, currFeature])
    #         shuffle_acc                    = winModel.score( x_Test_Temp,y_Test )
    #         #prediction_values = winModel.predict(x_Test_Temp)
    #         #shuffle_acc = np.mean((y_Test - prediction_values) * (y_Test - prediction_values))
    #         arrImportance2[currFeature,exp]= ( avgTrainR2Score-shuffle_acc ) /avgTrainR2Score
    #     print('{0:4} ==> {1:4d}'.format("Method 2 Exp Number", exp+1))
    # featureImportanceM2 = np.abs(np.mean( arrImportance2, axis=1 ))
    # minmax = MinMaxScaler()
    # featureImportanceM2 = minmax.fit_transform(1 * np.array([featureImportanceM2]).T).T[0]
    # featureNameValue = np.array(zip(featureName, featureImportanceM2),dtype=[('Feature', 'S5'), ('Importance', float)])
    # featureNameValue[::-1].sort(order='Importance')  # Sort the features by value
    # separateFeatureNameValue = zip(*featureNameValue)  # this will be used to plot apply list
    # ToolGrafic.saveFeatureImportanceScore( list(separateFeatureNameValue[0]), list(separateFeatureNameValue[1]),
    #                                        SIMULATION_PATH_GA, 'M2', numberOfFeatures, varColor='coral' )
    #
    # ToolGrafic.saveFeatureImportanceScoreByType( list(separateFeatureNameValue[0]), list(separateFeatureNameValue[1]),
    #                                              SIMULATION_PATH_GA, 'M2', len(featureName),
    #                                              colorByFeatureType, featureByType )
    #
    # dfFeaturesScores = pd.concat([ pd.DataFrame(list(separateFeatureNameValue[0])),
    #                                pd.DataFrame(list(separateFeatureNameValue[1]))], axis=1, ignore_index=True)
    # dfFeaturesScores.to_csv(os.path.join(os.path.sep, SIMULATION_PATH_GA, "M2_FeatureImportance.csv"))

def getImportanceMethod3( dfDatabase, numberOfFeatures, featureName, minNumberOfNeuronsByLayer,
                          maxNumberOfNeuronsByLayer,
                          maxNumOfExp, maxNumOfTrials, trainFactor,
                          colorByFeatureType, featureByType,
                          SIMULATION_PATH_GA,  individual=[]):
    # actFunct     = ""
    arrImportance2 = np.empty((numberOfFeatures, maxNumOfExp))
    for exp in range(0, maxNumOfExp):
        x_Train, y_Train, x_Test, y_Test, meanTarget, stdTarget = getScaledDatabase(dfDatabase, trainFactor)
        Y_Test_deNorm = getVarDesnormalization(y_Test, meanTarget, stdTarget)
        #for ni in range(2, maxNumberOfNeuronsByLayer + 1):
        for ni in range(2, 9):
            arq      = (ni,)
            actFunct = 'tanh'
            for i in range(0, maxNumOfTrials):
                mlpModel = MLPRegressor( solver='adam', hidden_layer_sizes=arq, max_iter=1500, tol=1e-12,
                                         verbose=False,
                                         shuffle=True, activation=actFunct, early_stopping=True, validation_fraction=0.10 )
                mlpModel.fit(x_Train, y_Train)
                y_predicted        = mlpModel.predict(x_Test)
                y_predicted_deNorm = getVarDesnormalization(y_predicted, meanTarget, stdTarget)
                currError = np.mean((Y_Test_deNorm-y_predicted_deNorm)*(Y_Test_deNorm-y_predicted_deNorm))
                if ((ni == 2) and (i == 0)):
                    winError = currError
                    winModel = copy.deepcopy(mlpModel)
                else:
                    if (currError< winError):
                        winError = currError
                        winModel = copy.deepcopy(mlpModel)
        for currFeature in range(0, numberOfFeatures):
            x_Test_Temp            = copy.deepcopy(x_Test)
            np.random.shuffle(x_Test_Temp[:, currFeature])
            y_predicted            = winModel.predict(x_Test_Temp)
            y_predicted_deNorm     = getVarDesnormalization(y_predicted, meanTarget, stdTarget)
            currError              = np.mean((Y_Test_deNorm-y_predicted_deNorm)*(Y_Test_deNorm-y_predicted_deNorm))
            arrImportance2[currFeature, exp] = ( (winError - currError)) / winError
        print('Method 3...' + '  ' + '{0:4} ==> {1:4d}'.format("Exp Number", exp))
    featureImportanceM2 = np.abs(np.mean(arrImportance2, axis=1))
    minmax = MinMaxScaler()
    featureImportanceM2 = minmax.fit_transform(1 * np.array([featureImportanceM2]).T).T[0]
    featureNameValue = np.array(zip(featureName, featureImportanceM2), dtype=[('Feature', 'S5'), ('Importance', float)])
    featureNameValue[::-1].sort(order='Importance')  # Sort the features by value
    separateFeatureNameValue = zip(*featureNameValue)  # this will be used to plot apply list
    ToolGrafic.saveFeatureImportanceScore(list(separateFeatureNameValue[0]), list(separateFeatureNameValue[1]),
                                          SIMULATION_PATH_GA, 'M3', numberOfFeatures, varColor='coral')

    ToolGrafic.saveFeatureImportanceScoreByType(list(separateFeatureNameValue[0]), list(separateFeatureNameValue[1]),
                                                SIMULATION_PATH_GA, 'M3', len(featureName),
                                                colorByFeatureType, featureByType)

    dfFeaturesScores = pd.concat([pd.DataFrame(list(separateFeatureNameValue[0])),
                                  pd.DataFrame(list(separateFeatureNameValue[1]))], axis=1, ignore_index=True)
    dfFeaturesScores.to_csv(os.path.join(os.path.sep, SIMULATION_PATH_GA, "M3_FeatureImportance.csv"))
    varAnalysisByMethod_3( dfDatabase, list(separateFeatureNameValue[0]), featureName, maxNumberOfNeuronsByLayer=8,
                           maxNumOfExp=1, maxNumOfTrials=1, trainFactor=trainFactor,
                           method="M3", SIMULATION_PATH_GA=SIMULATION_PATH_GA)

    # actFunct = ""
    # for exp in range(0, maxNumOfExp):
    #     x_Train, y_Train, x_Test, y_Test, meanTarget, stdTarget = getScaledDatabase(dfDatabase, trainFactor)
    #     avgTrainR3Score = 0.0
    #     for i in range(0, maxNumOfTrials):
    #         geneArq_1 = random.randint(minNumberOfNeuronsByLayer, maxNumberOfNeuronsByLayer)
    #         geneArq_2 = random.randint(minNumberOfNeuronsByLayer, maxNumberOfNeuronsByLayer)
    #         geneArq_3 = random.randint(31, 33)
    #         if (geneArq_3 == 31):
    #             actFunct = 'logistic'
    #         elif (geneArq_3 == 32):
    #             actFunct = 'tanh'
    #         elif (geneArq_3 == 33):
    #             actFunct = 'relu'
    #         mlpModel = MLPRegressor(solver='adam', hidden_layer_sizes=(geneArq_1, geneArq_2), max_iter=1500, tol=1e-12,
    #                                 verbose=False,
    #                                 shuffle=True, activation=actFunct, early_stopping=True, validation_fraction=0.10)
    #         mlpModel.fit(x_Train, y_Train)
    #         predicted_values = mlpModel.predict(x_Test)
    #         currR3Score = np.mean((y_Test - predicted_values) * (y_Test - predicted_values))
    #         if (i == 0):
    #             winR3Score = currR3Score
    #             winModel = copy.deepcopy(mlpModel)
    #         else:
    #             if (currR3Score < winR3Score):
    #                 winR3Score = currR3Score
    #                 winModel = copy.deepcopy(mlpModel)
    #         avgTrainR3Score = avgTrainR3Score + currR3Score
    #     avgTrainR3Score = avgTrainR3Score / maxNumOfTrials
    #     if exp == 0:
    #         arrImportance3 = np.empty((numberOfFeatures, maxNumOfExp))
    #     for currFeature in range(0, numberOfFeatures):
    #         x_Test_Temp = copy.deepcopy(x_Test)
    #         np.random.shuffle(x_Test_Temp[:, currFeature])
    #         prediction_values = winModel.predict(x_Test_Temp)
    #         shuffle_acc = np.mean((y_Test - prediction_values) * (y_Test - prediction_values))
    #         arrImportance3[currFeature, exp] = (avgTrainR3Score - shuffle_acc) / avgTrainR3Score
    #     print('{0:4} ==> {1:4d}'.format("Method 3 Exp Number", exp+1))
    # featureImportanceM3 = np.abs(np.mean(arrImportance3, axis=1))
    # minmax = MinMaxScaler()
    # featureImportanceM3 = minmax.fit_transform(1 * np.array([featureImportanceM3]).T).T[0]
    # featureNameValue = np.array(zip(featureName, featureImportanceM3), dtype=[('Feature', 'S5'), ('Importance', float)])
    # featureNameValue[::-1].sort(order='Importance')  # Sort the features by value
    # separateFeatureNameValue = zip(*featureNameValue)  # this will be used to plot apply list
    # ToolGrafic.saveFeatureImportanceScore( list(separateFeatureNameValue[0]), list(separateFeatureNameValue[1]),
    #                                        SIMULATION_PATH_GA, 'M3', numberOfFeatures, varColor='blue' )
    #
    # ToolGrafic.saveFeatureImportanceScoreByType( list(separateFeatureNameValue[0]), list(separateFeatureNameValue[1]),
    #                                              SIMULATION_PATH_GA,'M3',numberOfFeatures,colorByFeatureType,featureByType )
    # dfFeaturesScores = pd.concat(
    #     [pd.DataFrame(list(separateFeatureNameValue[0])), pd.DataFrame(list(separateFeatureNameValue[1]))], axis=1, ignore_index=True)
    # dfFeaturesScores.to_csv(os.path.join(os.path.sep, SIMULATION_PATH_GA, "M3_FeatureImportance.csv"))
def filteredMethodsCalculation( dfDataBase,  numberOfFeatures, featureName, SIMULATION_PATH_GA ):
    X, Y, _, _, _, _ = getScaledDatabase( dfDataBase, 1 )
    ranks = {}
    lr                  = LinearRegression(normalize=True)
    lr.fit(X, Y)
    ranks["Linear reg"] = rank_to_dict(np.abs(lr.coef_), featureName)

    ridge           = Ridge(alpha=7)
    ridge.fit(X, Y)
    ranks["Ridge"]  = rank_to_dict(np.abs(ridge.coef_), featureName)

    lasso           = Lasso(alpha=.05)
    lasso.fit(X, Y)
    ranks["Lasso"]  = rank_to_dict(np.abs(lasso.coef_), featureName)

    rlasso              = RandomizedLasso(max_iter=1000)
    rlasso.fit(X, Y)
    ranks["Stability"]  = rank_to_dict(np.abs(rlasso.scores_), featureName)

    # stop the search when 5 features are left (they will get equal scores)
    rfe             = RFE(lr, n_features_to_select=1)
    rfe.fit(X, Y)
    ranks["RFE"]    = rank_to_dict(map(float, rfe.ranking_), featureName, order=-1)

    rf          = RandomForestRegressor()
    rf.fit(X, Y)
    ranks["RF"] = rank_to_dict(rf.feature_importances_, featureName)

    f, pval         = f_regression(X, Y, center=True)
    ranks["Corr."]  = rank_to_dict(f, featureName)

    mine        = minepy.mine.MINE()
    mic_scores  = []
    for i in range(X.shape[1]):
        mine.compute_score(X[:, i], Y)
        m = mine.mic()
        mic_scores.append(m)

    ranks["MIC"] = rank_to_dict(mic_scores, featureName)

    r = {}
    for name in featureName:
        r[name] = round(np.mean([ranks[method][name] for method in ranks.keys()]), 2)

    methods = sorted( ranks.keys() )
    ranks["Mean"] = r
    methods.append( "Mean" )
    pdScores = pd.DataFrame()
    for method in methods:
        lst      = ranks[method]
        values   = pd.DataFrame( lst.values() )
        pdScores = pd.concat( [pdScores, values], axis=1, ignore_index=True )
    methods.insert( 0,"FEATURES" )
    pdScores             = pd.concat( [pd.DataFrame( list( lst ) ), pdScores], axis=1, ignore_index=True )
    pdScoreFinal         = pdScores.sort_values( by=[9], ascending=[False] )
    pdScoreFinal         = pdScoreFinal.reset_index( drop=True )
    pdScoreFinal.columns = methods

    pdScoreFinal.to_csv( os.path.join( os.path.sep, SIMULATION_PATH_GA, "Other_FeatureImportanceMethods.csv" ) )
    ToolGrafic.saveFeatureImportanceScore( list( pdScoreFinal.ix[:,0] ), list( pdScoreFinal.ix[:,pdScoreFinal.shape[1]-1] ),
                                           SIMULATION_PATH_GA, 'M_OTHER', len(featureName), varColor='cyan' )
    #ToolGrafic.saveFeatureImportanceScoreByType(list(separateFeatureNameValue[0]), list(separateFeatureNameValue[1]),
    #                                            SIMULATION_PATH_GA, method, len(featureName),
    #                                            colorByFeatureType, featureByType)

def varAnalysisByMethod_2 ( dfDatabase, methodFeatureRanking, featuresName, maxNumberOfNeuronsByLayer,
                                 maxNumOfExp, maxNumOfTrials, trainFactor, method, SIMULATION_PATH_GA):
    varList                = []
    varListR2AdjValue      = []
    varListR2Value         = []
    varListR2SpearmanValue = []
    varListRmse            = []
    vargroup               = 1
    for selFeature  in range( 0, numberOfFeatures-1 ):
        featureToTake =  methodFeatureRanking[0:selFeature+2]
        lstIdxFeature = []
        for currFeature in featureToTake:
            idx = 0
            for feature in featuresName:
                if ( currFeature==feature ):
                    lstIdxFeature.append(idx)
                    break
                idx = idx + 1
        lstR2AdjScore     = []
        lstR2Score        = []
        lstR2SpearmanScore= []
        lstListRmse       = []
        for exp in range( 0, maxNumOfExp ):
            x_Train, y_Train, x_Test, y_Test, meanTarget, stdTarget = getScaledDatabase(dfDatabase, trainFactor)
            x_Train       = x_Train[:,lstIdxFeature]
            x_Test        = x_Test[:, lstIdxFeature]
            Y_Test_deNorm = getVarDesnormalization( y_Test, meanTarget, stdTarget )
            for ni in range( 2, maxNumberOfNeuronsByLayer + 1 ):
                arq      = ( ni,)
                actFunct = 'tanh'
                for i in range( 0, maxNumOfTrials ):
                    mlpModel = MLPRegressor(solver='adam', hidden_layer_sizes=arq, max_iter=1500, tol=1e-12,
                                            verbose=False,
                                            shuffle=True, activation=actFunct, early_stopping=True,
                                            validation_fraction=0.10)
                    mlpModel.fit(x_Train, y_Train)
                    y_predicted         = mlpModel.predict(x_Test)
                    y_predicted_deNorm  = getVarDesnormalization(y_predicted, meanTarget, stdTarget)
                    _, _, r_value, _, _ = stats.linregress(Y_Test_deNorm, y_predicted_deNorm)
                    r2_adj              = 1 - ( 1-r_value*r_value )*( x_Train.shape[0]-1 )/( x_Train.shape[0]-len( featureToTake )-1 )
                    r2                  = r_value* r_value
                    statistics          = stats.spearmanr(Y_Test_deNorm, y_predicted_deNorm)
                    r_spearman          = statistics[0]
                    rmse                = rmsCalculation(Y_Test_deNorm, y_predicted_deNorm)
                    if ( ( exp==0 ) and ( ni == 2 ) and ( i == 0 ) ):
                        winR2AdjScore = r2_adj
                        winR2Score    = r2
                        winRSpearman  = r_spearman
                        winRmse       = rmse
                    else:
                        if ( r2_adj > winR2AdjScore ):
                            winR2Score = r2_adj
                        if ( r2 > winR2Score ):
                            winR2Score = r2
                        if ( r_spearman> winRSpearman ):
                            winRSpearman = r_spearman
                        if ( rmse<winRmse ):
                            winRmse = rmse
            lstR2AdjScore.append( winR2AdjScore )
            lstR2Score.append( winR2Score )
            lstR2SpearmanScore.append(winRSpearman)
            lstListRmse.append( winRmse )
        varList.append( selFeature+2 )
        varListR2AdjValue.append( np.mean(lstR2AdjScore) )
        varListR2Value.append( np.mean(lstR2Score) )
        varListR2SpearmanValue.append( np.mean(lstR2SpearmanScore) )
        varListRmse.append( np.mean( lstListRmse ) )
        print('Var Analysis Method 2...' + '  ' + '{0:4} ==> {1:4d}'.format("Group Number", vargroup))
        vargroup = vargroup + 1

    ToolGrafic.saveAnalysisByMethod( varList, varListR2Value, "R2", method, SIMULATION_PATH_GA)
    ToolGrafic.saveAnalysisByMethod( varList, varListR2AdjValue, "RA", method, SIMULATION_PATH_GA)
    ToolGrafic.saveAnalysisByMethod( varList, varListR2SpearmanValue, "RE", method, SIMULATION_PATH_GA)
    ToolGrafic.saveAnalysisByMethod( varList, varListRmse, "RMSE", method, SIMULATION_PATH_GA)

    ToolGrafic.saveAnalysisByMethod2( varList, varListR2AdjValue, varListR2Value, method, SIMULATION_PATH_GA )

    df = pd.concat([pd.DataFrame(varList),pd.DataFrame(varListR2AdjValue), pd.DataFrame(varListR2Value),
                    pd.DataFrame(varListR2SpearmanValue), pd.DataFrame(varListRmse)], ignore_index=True, axis=1)
    df.to_csv( os.path.join( os.path.sep, SIMULATION_PATH_GA, method +"_EPS_NumVarAnalysis.csv" ) )

def varAnalysisByMethod_3 ( dfDatabase, methodFeatureRanking, featuresName, maxNumberOfNeuronsByLayer,
                                 maxNumOfExp, maxNumOfTrials, trainFactor, method, SIMULATION_PATH_GA):
    varList                = []
    varError               = []
    vargroup               = 1
    for selFeature  in range( 0, numberOfFeatures-1 ):
        featureToTake =  methodFeatureRanking[0:selFeature+2]
        lstIdxFeature = []
        for currFeature in featureToTake:
            idx = 0
            for feature in featuresName:
                if ( currFeature==feature ):
                    lstIdxFeature.append(idx)
                    break
                idx = idx + 1
        lstError = []
        for exp in range( 0, maxNumOfExp ):
            x_Train, y_Train, x_Test, y_Test, meanTarget, stdTarget = getScaledDatabase(dfDatabase, trainFactor)
            x_Train       = x_Train[:,lstIdxFeature]
            x_Test        = x_Test[:, lstIdxFeature]
            Y_Test_deNorm = getVarDesnormalization( y_Test, meanTarget, stdTarget )
            for ni in range( 2, maxNumberOfNeuronsByLayer + 1 ):
                arq      = ( ni,)
                actFunct = 'tanh'
                for i in range( 0, maxNumOfTrials ):
                    mlpModel = MLPRegressor(solver='adam', hidden_layer_sizes=arq, max_iter=1500, tol=1e-12,
                                            verbose=False,
                                            shuffle=True, activation=actFunct, early_stopping=True,
                                            validation_fraction=0.10)
                    mlpModel.fit(x_Train, y_Train)
                    y_predicted         = mlpModel.predict(x_Test)
                    y_predicted_deNorm  = getVarDesnormalization(y_predicted, meanTarget, stdTarget)
                    lstError1            = list((Y_Test_deNorm-y_predicted_deNorm)*(Y_Test_deNorm-y_predicted_deNorm))
                    currError           = np.mean(lstError1)
                    #_, _, r_value, _, _ = stats.linregress(Y_Test_deNorm, y_predicted_deNorm)
                    #r2_adj              = 1 - ( 1-r_value*r_value )*( x_Train.shape[0]-1 )/( x_Train.shape[0]-len( featureToTake )-1 )
                    #r2                  = r_value* r_value
                    #statistics          = stats.spearmanr(Y_Test_deNorm, y_predicted_deNorm)
                    #r_spearman          = statistics[0]
                    #rmse                = rmsCalculation(Y_Test_deNorm, y_predicted_deNorm)
                    if ( ( exp==0 ) and ( ni == 2 ) and ( i == 0 ) ):
                        winError = currError
                    else:
                       if currError<winError:
                           winError = currError
            lstError.append( winError )
        varList.append( selFeature+2 )
        varError.append( np.mean( lstError ) )
        print('Var Analysis Method 3...' + '  ' + '{0:4} ==> {1:4d}'.format("Group Number", vargroup))
        vargroup = vargroup + 1
    ToolGrafic.saveAnalysisByMethod( varList, varError, "ERROR", method, SIMULATION_PATH_GA)
    #ToolGrafic.saveAnalysisByMethod2( varList, varListR2AdjValue, varListR2Value, method, SIMULATION_PATH_GA )
    df = pd.DataFrame( varError )
    df.to_csv(os.path.join(os.path.sep, SIMULATION_PATH_GA, method + "_EPS_NumVarAnalysis.csv"))
    #df = pd.concat([pd.DataFrame(varList),pd.DataFrame(varListR2AdjValue), pd.DataFrame(varListR2Value),
    #                pd.DataFrame(varListR2SpearmanValue), pd.DataFrame(varListRmse)], ignore_index=True, axis=1)
    #f.to_csv( os.path.join( os.path.sep, SIMULATION_PATH_GA, method +"_EPS_NumVarAnalysis.csv" ) )

# def executeGa( maxNumGenerations, maxNumExperiments, populationSize,
#                numberOfFeatures, tournamentSize, minSteadyStatedPercentage,
#                maxSteadyStatedPercentage, stepToChangeNewIndNum,numIndForImpor,
#                featureName, probCross, probMut, databaseName,
#                trainFactor, colorByFeatureType, featureByType, lstScoreNames,
#                SIMULATION_PATH, SIMULATION_PATH_GA, SIMULATION_PATH_NN ):
#
#     databaseFileName                 = os.path.join( os.path.sep, SIMULATION_PATH, databaseName)
#     dfDataBase                       = pd.read_csv(databaseFileName)
#     #arrDataBase                      = dfDataBase.values
#     print("Feature Importance Calculation  by Method 2....!!!\n")
#     getImportanceMethod2( dfDataBase, numberOfFeatures, featureName, minNumberOfNeuronsByLayer=5,
#                           maxNumberOfNeuronsByLayer=8,
#                           maxNumOfExp=1, maxNumOfTrials=1, trainFactor=0.9,
#                           colorByFeatureType=colorByFeatureType, featureByType=featureByType,
#                           SIMULATION_PATH_GA=SIMULATION_PATH_GA, individual=[] )
#     print("Feature Importance Method 2 Finish....!!!\n")
#
#     print("Feature Importance Calculation  by Method 3....!!!\n")
#     getImportanceMethod3( dfDataBase, numberOfFeatures, featureName, minNumberOfNeuronsByLayer=5,
#                           maxNumberOfNeuronsByLayer=8,
#                           maxNumOfExp=1, maxNumOfTrials=1, trainFactor=0.9,
#                           colorByFeatureType=colorByFeatureType, featureByType=featureByType,
#                           SIMULATION_PATH_GA=SIMULATION_PATH_GA, individual=[] )
#     print("Feature Importance Method 3 Finish....!!!\n")
#     #data[:, [1, 9]] specific columns in array
#     print("Feature Importance Calculation  by Other Methods ....!!!\n")
#     filteredMethodsCalculation(dfDataBase, numberOfFeatures, featureName, SIMULATION_PATH_GA)
#     print("Feature Importance  by Other Methods Finish....!!!\n")
#     dfOtherMethods = pd.read_csv(os.path.join(SIMULATION_PATH_GA,'Other_FeatureImportanceMethods.csv'))
#     lstColName = list(dfOtherMethods.columns)
#     lstColName = lstColName[1:len(lstColName)-1]
#     dfM3 = pd.read_csv(os.path.join(SIMULATION_PATH_GA, 'M3_FeatureImportance.csv'))
#     dfM2 = pd.read_csv(os.path.join(SIMULATION_PATH_GA, 'M2_FeatureImportance.csv'))
#     lstFeaturesOM = list(dfOtherMethods.ix[:,1])
#     lstFeature    = list(dfM2.ix[:,1])
#     lstTemp       = []
#     for nameFeatureToFind in lstFeaturesOM:
#         row = 0
#         for nameFeature in lstFeature:
#             if nameFeatureToFind==nameFeature:
#                lstTemp.append(dfM2.ix[row,2])
#                break
#             row = row +1
#     dfNew =  dfOtherMethods.ix[:,1:dfOtherMethods.shape[1]-1]
#     dfNew =  pd.concat([dfNew, pd.DataFrame(lstTemp)], axis=1, ignore_index=True)
#     lstColName.append("SEVM2")
#     lstFeature = list(dfM3.ix[:, 1])
#     lstTemp = []
#     for nameFeatureToFind in lstFeaturesOM:
#         row = 0
#         for nameFeature in lstFeature:
#             if nameFeatureToFind == nameFeature:
#                 lstTemp.append(dfM3.ix[row, 2])
#                 break
#             row = row + 1
#     dfNew = pd.concat([dfNew, pd.DataFrame(lstTemp)], axis=1, ignore_index=True)
#     lstColName.append("SEVM3")
#     lstColName.append("Mean")
#     avg = pd.DataFrame( dfNew.mean(axis=1) )
#     dfNew = pd.concat([dfNew, avg], axis=1, ignore_index=True)
#     dfNew = dfNew.sort_values(by=[11], ascending=[False])
#     dfNew = dfNew.reset_index(drop=True)
#     dfNew.columns= lstColName
#     dfNew.to_csv(os.path.join(os.path.sep, SIMULATION_PATH_GA, "PartialConsensusFeatureImportance.csv"))
#
#     ToolGrafic.saveFeatureImportanceScore( list(dfNew.ix[:, 0]),
#                                            list(dfNew.ix[:, dfNew.shape[1] - 1]),
#                                            SIMULATION_PATH_GA, 'PartialConsensus', len(featureName), varColor='y')
#
#     ToolGrafic.saveFeatureImportanceScoreByType( list(dfNew.ix[:, 0]), list(dfNew.ix[:, dfNew.shape[1] - 1]),
#                                                  SIMULATION_PATH_GA,'PartialConsensus', len(featureName), colorByFeatureType,
#                                                  featureByType )
#
#
#
#     creator.create("FitnessMin", base.Fitness, weights=(-1.0, ))
#     creator.create("Individual", list, fitness=creator.FitnessMin)
#     # region GA Configuration
#     toolbox = base.Toolbox()
#     toolbox.register("attr_bool", random.randint, 0, 1)
#     toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=numberOfFeatures)
#     toolbox.register("population", tools.initRepeat, list, toolbox.individual)
#
#     #toolbox.register("mate", tools.cxTwoPoint)
#     toolbox.register("mate", tools.cxOrdered )
#     #toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
#     toolbox.register("mutate", tools.mutUniformInt, low=0, up=1, indpb=0.1)
#
#     toolbox.decorate("population",checkActiveFeatures3(2))
#
#     toolbox.decorate("mate", checkActiveFeatures(2))
#     toolbox.decorate("mutate", checkActiveFeatures(2))
#     toolbox.register("select", tools.selTournament, tournsize=tournamentSize)
#     toolbox.register("map", futures.map)
#     # endregion
#     arrFitnessExperiment   = np.empty((maxNumExperiments, maxNumGenerations ))
#     arrStdExperiment       = np.empty((maxNumExperiments, maxNumGenerations))
#     arrBestExperiment      = np.empty((maxNumGenerations,(numberOfFeatures+3)*maxNumExperiments))
#     for exp in range (1, maxNumExperiments+1):
#         x_Train, y_Train, x_Test, y_Test, meanTarget, stdTarget = getScaledDatabase(dfDataBase, trainFactor)
#         toolbox.register("evaluate", getFitness2, x_Train, y_Train, x_Test, y_Test,
#                          meanTarget, stdTarget, minNumberOfNeuronsByLayer, maxNumberOfNeuronsByLayer)
#         arrBestIndByGeneration = np.empty((maxNumGenerations, numberOfFeatures+3))
#         # region Initial Population
#         population = toolbox.population(n=populationSize)
#         # Evaluate the entire population
#         fitness = list(map(toolbox.evaluate, population))
#         for fit, ind in zip(fitness, population):
#             ind.fitness.values = (fit[0],)
#         # endregion
#         for gen in range(1, maxNumGenerations+1):
#             # region Operator Selection
#             toolbox.unregister("mate")
#             toolbox.unregister("mutate")
#             crossOp = random.randint(1,4)
#             if crossOp==1:
#                 toolbox.register( "mate", tools.cxOnePoint )
#             elif(crossOp==2):
#                 toolbox.register( "mate", tools.cxTwoPoint )
#             elif(crossOp==3):
#                 toolbox.register( "mate", tools.cxUniform,indpb=0.3 )
#             elif(crossOp==4):
#                 toolbox.register( "mate", tools.cxOrdered )
#             mutOp   = random.randint(1, 3)
#             if mutOp==1:
#                 toolbox.register( "mutate", tools.mutShuffleIndexes, indpb=0.4 )
#             elif(mutOp==2):
#                 toolbox.register( "mutate", tools.mutFlipBit, indpb=0.6 )
#             elif(mutOp==3):
#                 toolbox.register( "mutate", tools.mutUniformInt, low=0, up=1, indpb=0.3 )
#             toolbox.decorate("mate", checkActiveFeatures(2))
#             toolbox.decorate("mutate", checkActiveFeatures(2))
#             # endregion
#             if ( gen==1 ):
#                 newIndividualsNumber = int( minSteadyStatedPercentage*populationSize )
#             else:
#                 if ( ( gen%stepToChangeNewIndNum )==0 ):
#                     newIndividualsNumber = getNewIndividualsNumber( maxNumGenerations, gen, minSteadyStatedPercentage,
#                                                                           maxSteadyStatedPercentage, populationSize )
#             offspring = algorithms.varOr( population, toolbox, lambda_=newIndividualsNumber, cxpb=0.9, mutpb=0.1 )
#             fits      = toolbox.map( toolbox.evaluate, offspring )
#             for fit, ind in zip( fits, offspring ):
#                 ind.fitness.values    =  (fit[0],)
#             population = tools.selBest(population+ offspring, k=populationSize)
#             # Gather all the fitness in one list and print the stats
#             fits2 = [ind.fitness.values[0] for ind in population]
#             std      = np.std(fits2)
#             best_ind = tools.selBest(population, 1)[0]
#             #print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))
#             arrFitnessExperiment[exp-1,gen-1]   = best_ind.fitness.values[0]
#             arrStdExperiment[exp-1,gen-1]       = std
#             arrBestIndByGeneration[gen-1,:]     = getIndWithArchitecture ( best_ind )
#             if ( ( gen % 5 )==0 ):
#                 #mean = np.mean(fits2)
#                 minimum = np.min(fits2)
#                 maximum = np.max(fits2)
#                 print('{0:4} ==> {1:4d}'.format("Exp Number", exp )+"  "+ '{0:4} ==> {1:4d}'.format("Gen Number", gen )
#                       +"  "+ '{0:4} ==> {1:4f}'.format("Best Fitness", best_ind.fitness.values[0] )
#                       +"  "+ '{0:4} ==> {1:4f}'.format("Std Fitness", std )
#                       +"  "+ '{0:4} ==> {1:4f}'.format("Min Fitness", minimum)
#                       +"  "+ '{0:4} ==> {1:4f}'.format("Max Fitness", maximum))
#             if ( gen == ( maxNumGenerations ) ):
#                 modelName = 'M1_NNMODEL_'+str( exp )+'_'+str( gen )+'.sav'
#                 pickle.dump( best_ind.nnModel, open( os.path.join(SIMULATION_PATH_NN, modelName ), 'wb' ) )
#                 if best_ind.geneArq_3==31:
#                     strActFun= 'logistic'
#                 elif(best_ind.geneArq_3==32):
#                     strActFun = 'tanh'
#                 elif (best_ind.geneArq_3 == 33):
#                     strActFun = 'relu'
#                 if ( exp==1 ):
#                     strNNTitle1 = 'GA Exp'+','+'GA Gen'+','+'Slope'+','+'Intercep'+','+'r_value'+','+'p_value'+','+'std_err'+','+ \
#                                   'r_square'+','+'r_adjusted'+','+'r_spearman'+','+'rho_Spearman'+','+'ActFunt'+','+'NeuNum1'+','+ \
#                                   'NeuNum2'+','+'RMSE' + '\n'
#                     f = open( os.path.join( os.path.sep,SIMULATION_PATH_NN , "summaryEvoNN.csv"), "a" )
#                     f.write( strNNTitle1 )
#                     f.close()
#                 f1 = open( os.path.join( os.path.sep,SIMULATION_PATH_NN , "summaryEvoNN.csv"), "a" )
#                 strNNData =  str( exp )+','+str( gen )+','+str( best_ind.ArrStats[0,0]  )+','+str( best_ind.ArrStats[0,1] )+','+\
#                              str( best_ind.ArrStats[0,2] )+','+str( best_ind.ArrStats[0,3] )+','+str( best_ind.ArrStats[0,4] )+','+\
#                              str( best_ind.ArrStats[0,5] )+','+str( best_ind.ArrStats[0,6] )+','+str( best_ind.ArrStats[0,7] )+','+\
#                              str( best_ind.ArrStats[0,8] )+','+strActFun+','+str(best_ind.geneArq_1)+','+\
#                              str(best_ind.geneArq_2)+','+str( best_ind.ArrStats[0,9] )+'\n'
#                 f1.write( strNNData )
#                 f1.close()
#                 NNEvolutionPlot(best_ind.lstLostCurve,np.array( best_ind.Y_predicted ), np.array( best_ind.Y_test ),  best_ind.ArrStats[0,5], best_ind.ArrStats[0,0],
#                                 best_ind.ArrStats[0,1], best_ind.ArrStats[0,6], best_ind.ArrStats[0,7], best_ind.ArrStats[0,8],
#                                 best_ind.ArrStats[0,9], exp, gen, SIMULATION_PATH_NN)
#
#         minCol = ( numberOfFeatures+3 )*( exp-1 )
#         maxCol = ( numberOfFeatures+3 )* exp
#         arrBestExperiment[:,minCol:maxCol] = arrBestIndByGeneration
#         if ( exp==1 ):
#             importByFrequency = arrBestIndByGeneration[maxNumGenerations-numIndForImpor:maxNumGenerations,0:numberOfFeatures]
#         else:
#             importByFrequency = np.vstack((importByFrequency, arrBestIndByGeneration[maxNumGenerations-numIndForImpor:maxNumGenerations,0:numberOfFeatures]) )
#
#     meanFitnessByGeneration = np.mean( arrFitnessExperiment, axis=0 )
#     meanStdExperiment       = np.mean( arrStdExperiment,     axis=0 )
#     meanFeatureImportance   = np.mean( importByFrequency,    axis=0 )
#     featureNameValue        = np.array(zip(featureName, meanFeatureImportance), dtype=[('Feature', 'S5'), ('Importance', float)])
#     featureNameValue[::-1].sort(order='Importance') #Sort the features by value
#     separateFeatureNameValue = zip(*featureNameValue) #this will be used to plot apply list
#     ToolGrafic.saveFeatureImportanceScore( list(separateFeatureNameValue[0]),list(separateFeatureNameValue[1]),
#                                            SIMULATION_PATH_GA,'M1',numberOfFeatures, varColor='green' )
#     ToolGrafic.saveFeatureImportanceScoreByType( list(separateFeatureNameValue[0]), list(separateFeatureNameValue[1]),
#                                                  SIMULATION_PATH_GA, 'M1', numberOfFeatures, colorByFeatureType,
#                                                  featureByType )
#     #ToolGrafic.saveFeatureImportanceScore( list(separateFeatureNameValue[0]), list(separateFeatureNameValue[1]),
#     #                                       SIMULATION_PATH_GA,'M1',int(0.30*numberOfFeatures), varColor='green' )
#     with open(os.path.join(SIMULATION_PATH_GA,'M1_FeatureImportance.csv' ), 'w') as f:
#         writer = csv.writer(f, delimiter=',', lineterminator='\n')
#         writer.writerows(featureNameValue)
#     np.savetxt(os.path.join(SIMULATION_PATH_GA,'BestIndByGenerationByExp.csv' ), arrBestExperiment, delimiter=",")
#     np.savetxt(os.path.join(SIMULATION_PATH_GA,'StatisticsEvolutionMatrix.csv' ),
#                np.vstack((arrFitnessExperiment,meanFitnessByGeneration, meanStdExperiment)).T, delimiter=",")
#     ToolGrafic.saveEvolutionProcess(meanFitnessByGeneration, SIMULATION_PATH_GA, type='Fitness' )
#     ToolGrafic.saveEvolutionProcess(meanStdExperiment, SIMULATION_PATH_GA, type='std' )
#
#     #Calculating the number of important features for the final neural network
#     #This number indicates the number of features that are activated in average in the best individuals of each experiment
#     finalNumberOfFeatures = int(np.mean(np.sum(importByFrequency.T, axis=0))) #This number is calculated based in numIndForImpor
#     lstIndex=[]
#     for i in range (len(separateFeatureNameValue[0])):
#         for j in range (featureName.size):
#            if separateFeatureNameValue[0][i]==featureName[j]:
#                lstIndex.append(j)
#     selectedFeatures = lstIndex[0:finalNumberOfFeatures] #this are the index that must be selected
#
#     lstColName = []
#     dfGAMethods = pd.read_csv(os.path.join(SIMULATION_PATH_GA, 'PartialConsensusFeatureImportance.csv'))
#     dfM1 = pd.read_csv(os.path.join(SIMULATION_PATH_GA, 'M1_FeatureImportance.csv'), header=None)
#     lstFeaturesOM = list(dfGAMethods.ix[:, 1])
#     lstFeature    = list(dfM1.ix[:, 0])
#     #lstFeature = list(dfM1.ix[:, 0])
#     lstColName = list(dfGAMethods.columns)
#     lstColName = lstColName[1:len(lstColName) - 1]
#     lstTemp = []
#     for nameFeatureToFind in lstFeaturesOM:
#         row = 0
#         for nameFeature in lstFeature:
#             if nameFeatureToFind == nameFeature:
#                 lstTemp.append(dfM1.ix[row, 1])
#                 break
#             row = row + 1
#     dfNew1 = dfGAMethods.ix[:, 1:dfGAMethods.shape[1] - 1]
#     dfNew1 = pd.concat([dfNew1, pd.DataFrame(lstTemp)], axis=1, ignore_index=True)
#     lstColName.append("GASEL")
#     avg1 = pd.DataFrame(dfNew1.mean(axis=1))
#     dfNew1 = pd.concat([dfNew1, avg1], axis=1, ignore_index=True)
#     dfNew1 = dfNew1.sort_values(by=[dfNew1.shape[1] - 1], ascending=[False])
#     dfNew1 = dfNew1.reset_index(drop=True)
#     lstColName.append("Mean")
#     dfNew1.columns = lstColName
#     dfNew1.to_csv(os.path.join(os.path.sep, SIMULATION_PATH_GA, "FinalConsensusFeatureImportance.csv"))
#
#     ToolGrafic.saveFeatureImportanceScore( list(dfNew1.ix[:, 0]),
#                                            list(dfNew1.ix[:, dfNew1.shape[1] - 1]),
#                                            SIMULATION_PATH_GA, 'FinalConsensus', len(featureName), varColor='y' )
#
#     ToolGrafic.saveFeatureImportanceScoreByType( list(dfNew1.ix[:, 0]), list(dfNew1.ix[:, dfNew1.shape[1] - 1]),
#                                                  SIMULATION_PATH_GA, 'FinalConsensus', len(featureName),
#                                                  colorByFeatureType,featureByType )
#     #Training Final Neural Network
#     finalNNTrain(x_Train, y_Train, x_Test, y_Test, minNumberOfNeuronsByLayer, maxNumberOfNeuronsByLayer, selectedFeatures, SIMULATION_PATH_NN) #TODO change scaled

def main():
    ROOT_PATH           = '/home/edwin/SIMULATIONS'
    SIMULATION_NAME1    = '4STATE'
    SIMULATION_NAME2    = '1CTF'
    DATABASENAME        = '1ctf_protein_features.csv'

    #shutil.rmtree(os.path.join(os.path.sep, ROOT_PATH, SIMULATION_NAME1, SIMULATION_NAME2), ignore_errors=True)
    SIMULATION_PATH_GA = os.path.join(os.path.sep, ROOT_PATH, SIMULATION_NAME1, SIMULATION_NAME2, 'GAIMPORTANCE')
    SIMULATION_PATH = os.path.join(os.path.sep, ROOT_PATH, SIMULATION_NAME1)
    #os.makedirs(SIMULATION_PATH_GA)
    SIMULATION_PATH_NN = os.path.join(os.path.sep, ROOT_PATH, SIMULATION_NAME1, SIMULATION_NAME2, 'NNMODEL')

    maxNumGenerations           = 500
    maxNumExperiments           = 30
    populationSize              = 500
    # numberOfFeatures = 37
    #tournamentSize = 2
    minSteadyStatedPercentage   = 0.05
    maxSteadyStatedPercentage   = 0.50
    stepToChangeNewIndNum       = 50
    probCross                   = 0.90
    probMut                     = 0.10
    numIndForImpor              = 10
    trainFactor                 = 0.90
    minNumberOfNeuronsByLayer   = 5
    maxNumberOfNeuronsByLayer   = 10
    boolWithMethods23           = True

    maxNumOfTrials              = 30

    featureName = np.array(['PPHO', 'PPHI', 'HCA', 'HCB', 'HCN', 'NH', 'NB', 'NE', 'NG', 'NI', 'NT', 'NS', 'NO', 'RGY',
                            'AVCA', 'AVCB', 'AVPH', 'AVPS', 'NHBO', 'ENER', 'DOPE', 'DFIRE', 'GOAP', 'RWDD',
                            'RWOD', 'PLGSC', 'PMSUB', 'FRST', 'FRRAP', 'FRSOL', 'FRHYD', 'FRTOR', 'CHVOL',
                            'ANUA1', 'ANUA2', 'ANUA3', 'ANUA4'])

    colorByFeatureType = {1: '#92D050', 2: '#EEB5AA', 3: '#FFC000', 4: '#6699FF', 5: '#A03070'}

    featureByType = {'PPHO': 1, 'PPHI': 1, 'HCA': 1, 'HCB': 1, 'HCN': 1,
                     'NH': 2, 'NB': 2, 'NE': 2, 'NG': 2, 'NI': 2, 'NT': 2, 'NS': 2, 'NO': 2,
                     'RGY': 3, 'AVCA': 3, 'AVCB': 3, 'AVPH': 3, 'AVPS': 3, 'NHBO': 3,
                     'ENER': 4, 'DOPE': 4, 'DFIRE': 4, 'GOAP': 4, 'RWDD': 4,
                     'RWOD': 4, 'PLGSC': 4, 'PMSUB': 4, 'FRST': 4, 'FRRAP': 4, 'FRSOL': 4, 'FRHYD': 4, 'FRTOR': 4,
                     'CHVOL': 3, 'ANUA1': 3, 'ANUA2': 3, 'ANUA3': 3, 'ANUA4': 3}

    #lstScoreNames = ['HCN', 'GOAP', 'HCA', 'NO', 'NH', 'ANUA3', 'NHBO', 'FRRAP', 'NB', 'DFIRE', 'ANUA2',
    #                 'AVCA', 'RGY', 'PPHO', 'DOPE', 'PLGSC', 'NE', 'NS', 'NG', 'FRHYD', 'ANUA4', 'RWOD',
    #                 'NI', 'RWDD', 'AVCB', 'PMSUB', 'HCB', 'NT', 'ENER', 'AVPH', 'FRSOL', 'PPHI', 'CHVOL',
    #                 'ANUA1', 'FRTOR', 'FRST', 'AVPS']

    # start_time = time.time()
    # executeGa( maxNumGenerations, maxNumExperiments, populationSize,
    #           numberOfFeatures, tournamentSize, minSteadyStatedPercentage,
    #           maxSteadyStatedPercentage, stepToChangeNewIndNum, numIndForImpor,
    #           featureName,  probCross, probMut, DATABASENAME, trainFactor,
    #           colorByFeatureType, featureByType, lstScoreNames,
    #           SIMULATION_PATH, SIMULATION_PATH_GA, SIMULATION_PATH_NN )
    # print("--- %s seconds ---" % (time.time() - start_time))
    databaseFileName = os.path.join(os.path.sep, SIMULATION_PATH, DATABASENAME)
    dfDataBase = pd.read_csv(databaseFileName)
    # arrDataBase                      = dfDataBase.values
    if (boolWithMethods23):
        shutil.rmtree(os.path.join(os.path.sep, ROOT_PATH, SIMULATION_NAME1, SIMULATION_NAME2), ignore_errors=True)
        os.makedirs(SIMULATION_PATH_GA)
        os.makedirs(SIMULATION_PATH_NN)
        print("Feature Importance Calculation  by Method 2....!!!\n")
        #getImportanceMethod2(dfDataBase, numberOfFeatures, featureName, minNumberOfNeuronsByLayer=5,
         #                    maxNumberOfNeuronsByLayer=8,
         #                    maxNumOfExp=10,maxNumOfTrials=10, trainFactor=0.9,
         #                    colorByFeatureType=colorByFeatureType, featureByType=featureByType,
         #                   SIMULATION_PATH_GA=SIMULATION_PATH_GA, individual=[])
        method = 'M2'
        pythonFileNamePath = os.path.join(os.path.sep, os.getcwd(), 'ToolPMethod2.py')
        os.system('python ' + pythonFileNamePath + ' ' + ROOT_PATH + ' ' + SIMULATION_NAME1 + ' ' +
                  SIMULATION_NAME2 + ' ' + DATABASENAME + ' ' + str(numberOfFeatures) + ' ' +
                  str(maxNumExperiments) + ' ' + str(maxNumOfTrials) + ' ' + str(trainFactor) + ' ' + method)
        dfMethodFeatureRanking = pd.read_csv(os.path.join(SIMULATION_PATH_GA, 'M2_FeatureImportance.csv'))
        lstMethodFeatureRanking = list(dfMethodFeatureRanking.ix[:,1])
        print("Finding Optimal Number of features by Method 2....!!!\n")
        pythonFileNamePath2 = os.path.join(os.path.sep, os.getcwd(), 'ToolPVarAnalysisByMethod_2.py')
        lstFeatureName = list(featureName)
        strFeaturesName = ','.join([str(x) for x in lstFeatureName])
        strMethodFeatureRanking2 = ','.join([str(x) for x in lstMethodFeatureRanking])
        os.system('python ' + pythonFileNamePath2 + ' ' + ROOT_PATH + ' ' + SIMULATION_NAME1 + ' ' +
                  SIMULATION_NAME2 + ' ' + DATABASENAME + ' ' + strMethodFeatureRanking2 + ' ' + strFeaturesName + ' ' +
                  str(numberOfFeatures) + ' ' + str(maxNumberOfNeuronsByLayer) + ' ' + str(maxNumExperiments) + ' ' + str(maxNumOfTrials) + ' ' +
                  str(trainFactor) + ' ' + method)
        print("Feature Importance Method 2 Finish....!!!\n")

        print("Feature Importance Calculation  by Method 3....!!!\n")
        method = "M3"

        pythonFileNamePath = os.path.join(os.path.sep, os.getcwd(), 'ToolPMethod3.py')
        os.system('python ' + pythonFileNamePath + ' ' + ROOT_PATH + ' ' + SIMULATION_NAME1 + ' ' +
                  SIMULATION_NAME2 + ' ' + DATABASENAME + ' ' + str(numberOfFeatures) + ' ' +
                  str(maxNumExperiments) + ' ' + str(maxNumOfTrials) + ' ' + str(trainFactor) + ' ' + method)

        dfMethodFeatureRanking = pd.read_csv(os.path.join(SIMULATION_PATH_GA, 'M3_FeatureImportance.csv'))
        lstMethodFeatureRanking = list(dfMethodFeatureRanking.ix[:, 1])
        print("Finding Optimal Number of features by Method 3....!!!\n")
        pythonFileNamePath3 = os.path.join(os.path.sep, os.getcwd(), 'ToolPVarAnalysisByMethod_3.py')
        lstFeatureName = list(featureName)
        strFeaturesName = ','.join([str(x) for x in lstFeatureName])
        strMethodFeatureRanking3 = ','.join([str(x) for x in lstMethodFeatureRanking])

        pythonFileNamePath3 = os.path.join(os.path.sep, os.getcwd(), 'ToolPVarAnalysisByMethod_3.py')
        os.system('python ' + pythonFileNamePath3 + ' ' + ROOT_PATH + ' ' + SIMULATION_NAME1 + ' ' +
                  SIMULATION_NAME2 + ' ' + DATABASENAME + ' ' + strMethodFeatureRanking3 + ' ' + strFeaturesName + ' ' +
                  str(numberOfFeatures) + ' ' + str(maxNumberOfNeuronsByLayer) + ' ' + str(maxNumExperiments) + ' ' + str(maxNumOfTrials) + ' ' +
                  str(trainFactor) + ' ' + method)
        #getImportanceMethod3(dfDataBase, numberOfFeatures, featureName, minNumberOfNeuronsByLayer=5,
        #                     maxNumberOfNeuronsByLayer=8,
        #                     maxNumOfExp=1, maxNumOfTrials=1, trainFactor=0.9,
        #                     colorByFeatureType=colorByFeatureType, featureByType=featureByType,
        #                     SIMULATION_PATH_GA=SIMULATION_PATH_GA, individual=[])
        print("Feature Importance Method 3 Finish....!!!\n")
        # data[:, [1, 9]] specific columns in array
        print("Feature Importance Calculation  by Other Methods ....!!!\n")
        filteredMethodsCalculation(dfDataBase, numberOfFeatures, featureName, SIMULATION_PATH_GA)
        print("Feature Importance  by Other Methods Finish....!!!\n")
        dfOtherMethods = pd.read_csv(os.path.join(SIMULATION_PATH_GA, 'Other_FeatureImportanceMethods.csv'))
        lstColName = list(dfOtherMethods.columns)
        lstColName = lstColName[1:len(lstColName) - 1]
        dfM3 = pd.read_csv(os.path.join(SIMULATION_PATH_GA, 'M3_FeatureImportance.csv'))
        dfM2 = pd.read_csv(os.path.join(SIMULATION_PATH_GA, 'M2_FeatureImportance.csv'))
        lstFeaturesOM = list(dfOtherMethods.ix[:, 1])
        lstFeature = list(dfM2.ix[:, 1])
        lstTemp = []
        for nameFeatureToFind in lstFeaturesOM:
            row = 0
            for nameFeature in lstFeature:
                if nameFeatureToFind == nameFeature:
                    lstTemp.append(dfM2.ix[row, 2])
                    break
                row = row + 1
        dfNew = dfOtherMethods.ix[:, 1:dfOtherMethods.shape[1] - 1]
        dfNew = pd.concat([dfNew, pd.DataFrame(lstTemp)], axis=1, ignore_index=True)
        lstColName.append("SEVM2")
        lstFeature = list(dfM3.ix[:, 1])
        lstTemp = []
        for nameFeatureToFind in lstFeaturesOM:
            row = 0
            for nameFeature in lstFeature:
                if nameFeatureToFind == nameFeature:
                    lstTemp.append(dfM3.ix[row, 2])
                    break
                row = row + 1
        dfNew = pd.concat([dfNew, pd.DataFrame(lstTemp)], axis=1, ignore_index=True)
        lstColName.append("SEVM3")
        lstColName.append("Mean")
        avg = pd.DataFrame(dfNew.mean(axis=1))
        dfNew = pd.concat([dfNew, avg], axis=1, ignore_index=True)
        dfNew = dfNew.sort_values(by=[11], ascending=[False])
        dfNew = dfNew.reset_index(drop=True)
        dfNew.columns = lstColName
        dfNew.to_csv(os.path.join(os.path.sep, SIMULATION_PATH_GA, "PartialConsensusFeatureImportance.csv"))

        ToolGrafic.saveFeatureImportanceScore(list(dfNew.ix[:, 0]),
                                              list(dfNew.ix[:, dfNew.shape[1] - 1]),
                                              SIMULATION_PATH_GA, 'PartialConsensus', len(featureName), varColor='y')

        ToolGrafic.saveFeatureImportanceScoreByType(list(dfNew.ix[:, 0]), list(dfNew.ix[:, dfNew.shape[1] - 1]),
                                                    SIMULATION_PATH_GA, 'PartialConsensus', len(featureName), colorByFeatureType,
                                                    featureByType)
    else:
        shutil.rmtree(SIMULATION_PATH_NN, ignore_errors=True)
        os.makedirs(SIMULATION_PATH_NN)
    # creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    # creator.create("Individual", list, fitness=creator.FitnessMin)

    # from deap import creator, base
    # from some_module import set_creator

    # set_creator(creator)
    # creator.create("Fitness", base.Fitness, weights=(-1.0,))
    # creator.create("Individual", list, fitness=creator.Fitness)

    # region GA Configuration
    # toolbox = base.Toolbox()
    # toolbox.register("attr_bool", random.randint, 0, 1)
    # toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=numberOfFeatures)
    # toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # toolbox.register("mate", tools.cxTwoPoint)
    # toolbox.register("mate", tools.cxOrdered)
    # toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    # toolbox.register("mutate", tools.mutUniformInt, low=0, up=1, indpb=0.1)

    # toolbox.decorate("population", checkActiveFeatures3(2))

    # toolbox.decorate("mate", checkActiveFeatures(2))
    # toolbox.decorate("mutate", checkActiveFeatures(2))
    # toolbox.register("select", tools.selTournament, tournsize=tournamentSize)
    # toolbox.register("map", futures.map)
    # endregion
    print("Genetic Algorithm Initialization  by Method 1....!!!\n")
    arrFitnessExperiment = np.empty((maxNumExperiments, maxNumGenerations))
    arrStdExperiment = np.empty((maxNumExperiments, maxNumGenerations))
    arrBestExperiment = np.empty((maxNumGenerations, (numberOfFeatures + 3) * maxNumExperiments))
    for exp in range(1, maxNumExperiments + 1):
        x_Train, y_Train, x_Test, y_Test, meanTarget, stdTarget = getScaledDatabase(dfDataBase, trainFactor)
        toolbox.register("evaluate", getFitness2, x_Train, y_Train, x_Test, y_Test,
                         meanTarget, stdTarget, minNumberOfNeuronsByLayer, maxNumberOfNeuronsByLayer)
        arrBestIndByGeneration = np.empty((maxNumGenerations, numberOfFeatures + 3))
        # region Initial Population
        population = toolbox.population(n=populationSize)
        # Evaluate the entire population
        #fitness = list(toolbox.map(toolbox.evaluate, population))
        jobs    = toolbox.map(toolbox.evaluate, population)
        #fitness1 = jobs.get()
        #for fit, ind in zip(fitness1, population):
        #    ind.fitness.values       = (fit[0],)
        #    ind.fitness.AddGene1     = fit[1]
        #    ind.fitness.AddGene2     = fit[2]
        #    ind.fitness.AddGene3     = fit[3]
        #    ind.fitness.ArrStats     = fit[4]
        #    ind.fitness.lstLostCurve = fit[5]
        #    ind.fitness.nnModel      = fit[6]
        #    ind.fitness.Y_predicted  = fit[7]
        #    ind.fitness.Y_test       = fit[8]
        #fitness1 = jobs.get()
        for fit, ind in zip(jobs, population):
            ind.fitness.values = (fit[0],)
            ind.fitness.AddGene1 = fit[1]
            ind.fitness.AddGene2 = fit[2]
            ind.fitness.AddGene3 = fit[3]
            ind.fitness.ArrStats = fit[4]
            ind.fitness.lstLostCurve = fit[5]
            ind.fitness.nnModel = fit[6]
            ind.fitness.Y_predicted = fit[7]
            ind.fitness.Y_test = fit[8]


        # endregion
        for gen in range(1, maxNumGenerations + 1):
            # region Operator Selection
            toolbox.unregister("mate")
            toolbox.unregister("mutate")
            crossOp = random.randint(1, 4)
            if crossOp == 1:
                toolbox.register("mate", tools.cxOnePoint)
            elif (crossOp == 2):
                toolbox.register("mate", tools.cxTwoPoint)
            elif (crossOp == 3):
                toolbox.register("mate", tools.cxUniform, indpb=0.3)
            elif (crossOp == 4):
                toolbox.register("mate", tools.cxOrdered)
            mutOp = random.randint(1, 3)
            if mutOp == 1:
                toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.4)
            elif (mutOp == 2):
                toolbox.register("mutate", tools.mutFlipBit, indpb=0.6)
            elif (mutOp == 3):
                toolbox.register("mutate", tools.mutUniformInt, low=0, up=1, indpb=0.3)
            toolbox.decorate("mate", checkActiveFeatures(2))
            toolbox.decorate("mutate", checkActiveFeatures(2))
            # endregion
            if (gen == 1):
                newIndividualsNumber = int(minSteadyStatedPercentage * populationSize)
            else:
                if ((gen % stepToChangeNewIndNum) == 0):
                    newIndividualsNumber = getNewIndividualsNumber(maxNumGenerations, gen, minSteadyStatedPercentage,
                                                                   maxSteadyStatedPercentage, populationSize)
            offspring = algorithms.varOr(population, toolbox, lambda_=newIndividualsNumber, cxpb=probCross, mutpb=probMut)
            #fits = toolbox.map(toolbox.evaluate, offspring)
            jobs = toolbox.map(toolbox.evaluate, offspring)
            #fits = jobs.get()

            #for fit, ind in zip(fits, offspring):
            #    #ind.fitness.values = (fit[0],)
            #    ind.fitness.values          = (fit[0],)
            #    ind.fitness.AddGene1        = fit[1]
            #    ind.fitness.AddGene2        = fit[2]
            #    ind.fitness.AddGene3        = fit[3]
            #    ind.fitness.ArrStats        = fit[4]
            #    ind.fitness.lstLostCurve    = fit[5]
            #    ind.fitness.nnModel         = fit[6]
            #    ind.fitness.Y_predicted     = fit[7]
            #    ind.fitness.Y_test          = fit[8]
            for fit, ind in zip(jobs, offspring):
                #ind.fitness.values = (fit[0],)
                ind.fitness.values          = (fit[0],)
                ind.fitness.AddGene1        = fit[1]
                ind.fitness.AddGene2        = fit[2]
                ind.fitness.AddGene3        = fit[3]
                ind.fitness.ArrStats        = fit[4]
                ind.fitness.lstLostCurve    = fit[5]
                ind.fitness.nnModel         = fit[6]
                ind.fitness.Y_predicted     = fit[7]
                ind.fitness.Y_test          = fit[8]
            population = tools.selBest(population + offspring, k=populationSize)
            # Gather all the fitness in one list and print the stats
            fits2 = [ind.fitness.values[0] for ind in population]
            std = np.std(fits2)
            best_ind = tools.selBest(population, 1)[0]
            # print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))
            arrFitnessExperiment[exp - 1, gen - 1] = best_ind.fitness.values[0]
            arrStdExperiment[exp - 1, gen - 1] = std
            arrBestIndByGeneration[gen - 1, :] = getIndWithArchitecture(best_ind)
            if ((gen % 5) == 0):
                # mean = np.mean(fits2)
                minimum = np.min(fits2)
                maximum = np.max(fits2)
                print('{0:4} ==> {1:4d}'.format("Exp Number", exp) + "  " + '{0:4} ==> {1:4d}'.format("Gen Number", gen)
                      + "  " + '{0:4} ==> {1:4f}'.format("Best Fitness", best_ind.fitness.values[0])
                      + "  " + '{0:4} ==> {1:4f}'.format("Std Fitness", std)
                      + "  " + '{0:4} ==> {1:4f}'.format("Min Fitness", minimum)
                      + "  " + '{0:4} ==> {1:4f}'.format("Max Fitness", maximum))
            if (gen == (maxNumGenerations)):
                modelName = 'M1_NNMODEL_' + str(exp) + '_' + str(gen) + '.sav'
                pickle.dump(best_ind.fitness.nnModel, open(os.path.join(SIMULATION_PATH_NN, modelName), 'wb'))
                if best_ind.fitness.AddGene3 == 31:
                    strActFun = 'tanh'
                elif (best_ind.fitness.AddGene3 == 32):
                    strActFun = 'tanh'
                elif (best_ind.fitness.AddGene3 == 33):
                    strActFun = 'tanh'
                if (exp == 1):
                    strNNTitle1 = 'GA Exp' + ',' + 'GA Gen' + ',' + 'Slope' + ',' + 'Intercep' + ',' + 'r_value' + ',' + 'p_value' + ',' + 'std_err' + ',' + \
                                  'r_square' + ',' + 'r_adjusted' + ',' + 'r_spearman' + ',' + 'rho_Spearman' + ',' + 'ActFunt' + ',' + 'NeuNum1' + ',' + \
                                  'NeuNum2' + ',' + 'RMSE' + '\n'
                    f = open(os.path.join(os.path.sep, SIMULATION_PATH_NN, "summaryEvoNN.csv"), "a")
                    f.write(strNNTitle1)
                    f.close()
                f1 = open(os.path.join(os.path.sep, SIMULATION_PATH_NN, "summaryEvoNN.csv"), "a")
                strNNData = str(exp) + ',' + str(gen) + ',' + str(best_ind.fitness.ArrStats[0, 0]) + ',' + str(best_ind.fitness.ArrStats[0, 1]) + ',' + \
                            str(best_ind.fitness.ArrStats[0, 2]) + ',' + str(best_ind.fitness.ArrStats[0, 3]) + ',' + str(best_ind.fitness.ArrStats[0, 4]) + ',' + \
                            str(best_ind.fitness.ArrStats[0, 5]) + ',' + str(best_ind.fitness.ArrStats[0, 6]) + ',' + str(best_ind.fitness.ArrStats[0, 7]) + ',' + \
                            str(best_ind.fitness.ArrStats[0, 8]) + ',' + strActFun + ',' + str(best_ind.fitness.AddGene1) + ',' + \
                            str(best_ind.fitness.AddGene2) + ',' + str(best_ind.fitness.ArrStats[0, 9]) + '\n'
                f1.write(strNNData)
                f1.close()
                NNEvolutionPlot(best_ind.fitness.lstLostCurve, np.array(best_ind.fitness.Y_predicted), np.array(best_ind.fitness.Y_test), best_ind.fitness.ArrStats[0, 5], best_ind.fitness.ArrStats[0, 0],
                                best_ind.fitness.ArrStats[0, 1], best_ind.fitness.ArrStats[0, 6], best_ind.fitness.ArrStats[0, 7], best_ind.fitness.ArrStats[0, 8],
                                best_ind.fitness.ArrStats[0, 9], exp, gen, SIMULATION_PATH_NN)

        minCol = (numberOfFeatures + 3) * (exp - 1)
        maxCol = (numberOfFeatures + 3) * exp
        arrBestExperiment[:, minCol:maxCol] = arrBestIndByGeneration
        if (exp == 1):
            importByFrequency = arrBestIndByGeneration[maxNumGenerations - numIndForImpor:maxNumGenerations, 0:numberOfFeatures]
        else:
            importByFrequency = np.vstack((importByFrequency, arrBestIndByGeneration[maxNumGenerations - numIndForImpor:maxNumGenerations, 0:numberOfFeatures]))

    meanFitnessByGeneration = np.mean(arrFitnessExperiment, axis=0)
    meanStdExperiment = np.mean(arrStdExperiment, axis=0)
    meanFeatureImportance = np.mean(importByFrequency, axis=0)
    featureNameValue = np.array(zip(featureName, meanFeatureImportance), dtype=[('Feature', 'S5'), ('Importance', float)])
    featureNameValue[::-1].sort(order='Importance')  # Sort the features by value
    separateFeatureNameValue = zip(*featureNameValue)  # this will be used to plot apply list
    ToolGrafic.saveFeatureImportanceScore(list(separateFeatureNameValue[0]), list(separateFeatureNameValue[1]),
                                          SIMULATION_PATH_GA, 'M1', numberOfFeatures, varColor='green')
    ToolGrafic.saveFeatureImportanceScoreByType(list(separateFeatureNameValue[0]), list(separateFeatureNameValue[1]),
                                                SIMULATION_PATH_GA, 'M1', numberOfFeatures, colorByFeatureType,
                                                featureByType)
    # ToolGrafic.saveFeatureImportanceScore( list(separateFeatureNameValue[0]), list(separateFeatureNameValue[1]),
    #                                       SIMULATION_PATH_GA,'M1',int(0.30*numberOfFeatures), varColor='green' )
    with open(os.path.join(SIMULATION_PATH_GA, 'M1_FeatureImportance.csv'), 'w') as f:
        writer = csv.writer(f, delimiter=',', lineterminator='\n')
        writer.writerows(featureNameValue)
    np.savetxt(os.path.join(SIMULATION_PATH_GA, 'BestIndByGenerationByExp.csv'), arrBestExperiment, delimiter=",")
    np.savetxt(os.path.join(SIMULATION_PATH_GA, 'StatisticsEvolutionMatrix.csv'),
               np.vstack((arrFitnessExperiment, meanFitnessByGeneration, meanStdExperiment)).T, delimiter=",")
    ToolGrafic.saveEvolutionProcess(meanFitnessByGeneration, SIMULATION_PATH_GA, type='Fitness')
    ToolGrafic.saveEvolutionProcess(meanStdExperiment, SIMULATION_PATH_GA, type='std')

    # Calculating the number of important features for the final neural network
    # This number indicates the number of features that are activated in average in the best individuals of each experiment
    finalNumberOfFeatures = int(np.mean(np.sum(importByFrequency.T, axis=0)))  # This number is calculated based in numIndForImpor
    lstIndex = []
    for i in range(len(separateFeatureNameValue[0])):
        for j in range(featureName.size):
            if separateFeatureNameValue[0][i] == featureName[j]:
                lstIndex.append(j)
    selectedFeatures = lstIndex[0:finalNumberOfFeatures]  # this are the index that must be selected

    lstColName = []
    dfGAMethods = pd.read_csv(os.path.join(SIMULATION_PATH_GA, 'PartialConsensusFeatureImportance.csv'))
    dfM1 = pd.read_csv(os.path.join(SIMULATION_PATH_GA, 'M1_FeatureImportance.csv'), header=None)
    lstFeaturesOM = list(dfGAMethods.ix[:, 1])
    lstFeature = list(dfM1.ix[:, 0])
    # lstFeature = list(dfM1.ix[:, 0])
    lstColName = list(dfGAMethods.columns)
    lstColName = lstColName[1:len(lstColName) - 1]
    lstTemp = []
    for nameFeatureToFind in lstFeaturesOM:
        row = 0
        for nameFeature in lstFeature:
            if nameFeatureToFind == nameFeature:
                lstTemp.append(dfM1.ix[row, 1])
                break
            row = row + 1
    dfNew1 = dfGAMethods.ix[:, 1:dfGAMethods.shape[1] - 1]
    dfNew1 = pd.concat([dfNew1, pd.DataFrame(lstTemp)], axis=1, ignore_index=True)
    lstColName.append("GASEL")
    avg1 = pd.DataFrame(dfNew1.mean(axis=1))
    dfNew1 = pd.concat([dfNew1, avg1], axis=1, ignore_index=True)
    dfNew1 = dfNew1.sort_values(by=[dfNew1.shape[1] - 1], ascending=[False])
    dfNew1 = dfNew1.reset_index(drop=True)
    lstColName.append("Mean")
    dfNew1.columns = lstColName
    dfNew1.to_csv(os.path.join(os.path.sep, SIMULATION_PATH_GA, "FinalConsensusFeatureImportance.csv"))

    ToolGrafic.saveFeatureImportanceScore(list(dfNew1.ix[:, 0]),
                                          list(dfNew1.ix[:, dfNew1.shape[1] - 1]),
                                          SIMULATION_PATH_GA, 'FinalConsensus', len(featureName), varColor='y')

    ToolGrafic.saveFeatureImportanceScoreByType(list(dfNew1.ix[:, 0]), list(dfNew1.ix[:, dfNew1.shape[1] - 1]),
                                                SIMULATION_PATH_GA, 'FinalConsensus', len(featureName),
                                                colorByFeatureType, featureByType)
    # Training Final Neural Network
    finalNNTrain(x_Train, y_Train, x_Test, y_Test, minNumberOfNeuronsByLayer, maxNumberOfNeuronsByLayer, selectedFeatures, SIMULATION_PATH_NN)  # TODO change scaled
if __name__ == '__main__':
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    #toolbox.register('map', pool.map_async)
    toolbox.register('map', pool.map)
    tic = timer()
    main()
    pool.close()
    print timer() - tic