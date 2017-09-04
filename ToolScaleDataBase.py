import numpy as np
import os
import pandas as pd
import sklearn



def count_database_outlier(arr, mask, tolerance_std):
    """
    This function count the number of outliers of each feature using a mask and
    the tolerance_std parameters
    :param arr: This is the array that we need to clean
    :param mask: This is the mask used to select the feature that we will analise
    :param tolerance_std: This parameter is number of standard deviation used to determinate the range of the outliers
    :return:
    """
    nRow = arr.shape[0]
    nCol = arr.shape[1]
    list_outlier_number = [0] * nCol
    for col in range ( 0, nCol ):
        avg_feature = np.mean(arr[:, col])
        std_feature = np.std(arr[:, col])
        count_feature_outlier = 0
        for row in range (0, nRow ):
           if not(arr[row,col]<avg_feature + tolerance_std*std_feature) and (arr[row, col]> avg_feature-tolerance_std*std_feature):
               count_feature_outlier = count_feature_outlier+1
        list_outlier_number[col]=count_feature_outlier
    return list_outlier_number


def remove_database_outlier(arrToClean, featureMaskToCleanData, tolerance_std):
    """
    This function scan the outlier from each feature and then eliminate
    all that are ou of the range [mean+tolerance_std*std, mean-tolerance_std*std]
    :param arrToClean: np array to clean
    :param featureMaskToCleanData: This mask is used to select the features that the user selects to clean
    :param tolerance_std: This is the tolerance, and is used to determinate the interval where the outliers are.
    :return: This function do not return any value. arrToClean is an array that is update and then passed by reference
    """
    nRow            = arrToClean.shape[0]
    nCol            = arrToClean.shape[1]
    countByVar      = [0] * nCol
    for col in range(0,nCol):
        if ( featureMaskToCleanData[col]==1 ):
            stdCol  = np.std(arrToClean[:,col], axis=0)
            meanCol = np.mean(arrToClean[:,col], axis=0)
            maxLimit = meanCol + tolerance_std * stdCol
            minLimit = meanCol - tolerance_std * stdCol
            i = 0
            for row in range (0, nRow):
                tempValue =arrToClean[row, col] #dtToClean.ix[row, col]
                if not( (tempValue  < maxLimit) and
                     (tempValue > minLimit) ):
                    #idxListToRemove.append(row)
                    arrToClean[row, col] = meanCol
                    #dtToClean.ix[row, col]=meanCol
                    i = i + 1
            countByVar[col]=i



def scale_train_database(arrTrain, featureMask):
    """
    This function is used to scale each feature selected by featureMask in the train set
    :param arrTrain: This is the train database that will be scale
    :param featureMask: This is a vector of zeros and ones where 1 means that that feature is selected for scale.
    :return:
    """
    nRows = arrTrain.shape[0]
    nCols = arrTrain.shape[1]
    mean_df = []
    std_df  = []
    #mean_df = dfToScale.mean(axis=0)
    mean_df =  [0] * nCols
    std_df  =  [0] * nCols
    for col in range( 0,nCols ):
        if ( featureMask[col]==1 ):
            stdCol  = np.std(arrTrain[:,col], axis=0)
            meanCol = np.mean(arrTrain[:,col], axis=0)
            mean_df[col] = meanCol
            std_df[col]  = stdCol
            for row in range(0,nRows):
                arrTrain[row, col] = (arrTrain[row,col]-meanCol)/stdCol
    return mean_df, std_df


def scale_test_database(arrTest, mean_train, std_train, featureMask):
    """
    This function scale the test set using the mean and standard deviation from the train set using the mask to
    select the selected feature to scale
    :param arrTest: Array corresponding to the test set
    :param mean_train: List with the average value by feature in the train set
    :param std_train:  List with the standard deviation by feature in the test set
    :param featureMask: Boolean list where 1 indicates that the feature in that position will be scaled
    :return: The arrTrain is returned by reference
    """
    nRows = arrTest.shape[0]
    nCols = arrTest.shape[1]
    for col in range( 0, nCols ):
        if ( featureMask[col] == 1 and col!=nCols-1 ): #The target variable is not scaled
             arrTest[:,col] = (arrTest[:,col]-mean_train[col])/std_train[col]


def get_clean_database( dfDataBase, featureMaskOutlierClean, featureMaskScale,
                        trainFactor, tolerance_std, SIMULATION_PATH_GA ):

    """
    This is the main function has the pipeline to clean the data of the database provided
    :param dfDataBase: Database with all the examples
    :param featureMaskOutlierClean: Mask(1-0) used to select the feature where the cleaning process related to outliers
    :param featureMaskScale: Mask(1-0) used to select the feature that will be scaled
    :param trainFactor: factor that is used to divide the database in two sets Train and Set
    :param tolerance_std: Standard deviation factor used to determinate the range of the outliers
    :param SIMULATION_PATH_GA:  Path where the number of outliers by feature is saved.
    :return: x_Train(scaled), y_Train(scaled),
             x_Test(scaled), y_Test(Not scaled, the target value of the test set must be not scaled)
             When the scaled x_Test is used in the model the output is scaled, this output must be unscaled using
             mean_train and std_train and compared with x_Test
             mean_train(this value is used to scale the database), std_train(This value is used to scale the database)
    """

    dfDataBase = sklearn.utils.shuffle(dfDataBase)
    dfDataBase = dfDataBase.reset_index(drop=True)


    nRowsTrain = int ( dfDataBase.shape[0] * trainFactor )

    dfTrainSet = dfDataBase.ix[0:nRowsTrain,:]

    dfTestSet  = dfDataBase.ix[nRowsTrain:dfDataBase.shape[0],:]
    dfTestSet  = dfTestSet.reset_index(drop=True)


    arrTrain   = dfTrainSet.as_matrix()
    arrTest    = dfTestSet.as_matrix()


    df_outlier           = pd.DataFrame()
    train_number_outlier = list(count_database_outlier( arrTrain, featureMaskOutlierClean, tolerance_std ))
    test_number_outlier  = list(count_database_outlier( arrTest, featureMaskOutlierClean, tolerance_std ))
    df_outlier           = pd.concat( [df_outlier, pd.DataFrame( train_number_outlier ),
                                       pd.DataFrame( test_number_outlier ) ],
                                      axis=1, ignore_index=True )
    df_outlier.columns = ['Num Outlier Train', 'Num Outlier Test']
    df_outlier.to_csv( os.path.join(os.path.sep,SIMULATION_PATH_GA, 'outlier_dataset.csv') )

    remove_database_outlier(arrTrain, featureMaskOutlierClean, tolerance_std)

    mean_train, std_train = scale_train_database(arrTrain, featureMaskScale)

    scale_test_database(arrTest, mean_train, std_train, featureMaskScale)

    x_Train = arrTrain[:, 0:arrTrain.shape[1] - 1]
    y_Train = arrTrain[:, arrTrain.shape[1] - 1]

    x_Test = arrTest[:, 0:arrTest.shape[1] - 1] #The features are scaled with the mean and std of train set
    y_Test = arrTest[:, arrTest.shape[1] - 1]   #The target value in test set is not scaled


    return x_Train, y_Train, x_Test, y_Test, mean_train, std_train


# if __name__ == "__main__":
#     ROOT_PATH        = '/home/edwin/SIMULATIONS'
#     SIMULATION_NAME1 = '4STATE'
#     SIMULATION_NAME2 = '1ctf'
#     DATABASENAME     = '1ctf_protein_features.csv'
#
#     SIMULATION_PATH_GA = os.path.join(os.path.sep, ROOT_PATH, SIMULATION_NAME1, SIMULATION_NAME2, 'GAIMPORTANCE')
#     SIMULATION_PATH    = os.path.join(os.path.sep, ROOT_PATH, SIMULATION_NAME1)
#     # os.makedirs(SIMULATION_PATH_GA)
#     SIMULATION_PATH_NN = os.path.join(os.path.sep, ROOT_PATH, SIMULATION_NAME1, SIMULATION_NAME2, 'NNMODEL')
#
#     databaseFileName   = os.path.join(os.path.sep, SIMULATION_PATH, DATABASENAME)
#     dfDataBase         = pd.read_csv(databaseFileName)
#     #featureMaskToCleanData = [1,	1,	1,	1,	1,	0,	0,	0,
#     #                          0,	0,	0,	0,	0,	1,	1,	1,
#     #                          1,	1,	1,	1,	1,	1,	1,	1,
#     #                          1,	1,	1,	1,	1,	1,	1,	1,
#     #                          1,	1,	1,	1,	1,	1]
#
#     tolerance_std       = 3
#     featureMaskOutlierClean = [1, 1, 1, 1, 1, 0, 0, 0,
#                               0, 0, 0, 0, 0, 1, 1, 1,
#                               1, 1, 1, 1, 1, 1, 1, 1,
#                               1, 1, 1, 1, 1, 1, 1, 1,
#                               1, 1, 1, 1, 1, 0]
#
#     featureMaskScale       = [1, 1, 1, 1, 1, 1, 1, 1,
#                               1, 1, 1, 1, 1, 1, 1, 1,
#                               1, 1, 1, 1, 1, 1, 1, 1,
#                               1, 1, 1, 1, 1, 1, 1, 1,
#                               1, 1, 1, 1, 1, 1
#
#     ]
#
#     get_clean_database(dfDataBase, featureMaskOutlierClean, featureMaskScale, 0.90, tolerance_std, SIMULATION_PATH_GA)
#
