import os

ROOT_PATH        = '/home/edwin/SIMULATIONS'
SIMULATION_NAME1 = '4STATE'
SIMULATION_NAME2 = '1hdd-C'
DATABASENAME     = '1hdd-C_protein_features.csv'

numberOfFeatures          = 37
maxNumberOfNeuronsByLayer = 10
maxNumOfExp               = 2
maxNumOfTrials            = 10
trainFactor               = 0.9
method                    ="M2"

methodFeatureRanking =['AVCA', 'FRRAP', 'ANUA2', 'GOAP', 'FRTOR', 'NH', 'PPHI', 'DOPE', 'CHVOL', 'AVPH', 'FRST', 'RGY', 'ANUA1', 'PPHO',
    'NHBO', 'ENER', 'ANUA3', 'ANUA4', 'PLGSC', 'FRSOL', 'HCA', 'HCB', 'NE', 'PMSUB',
    'NB', 'NG', 'RWOD', 'HCN', 'NS', 'AVPS', 'AVCB', 'FRHYD', 'NO', 'RWDD', 'NT', 'DFIRE', 'NI']

featuresName= ['PPHO', 'PPHI', 'HCA', 'HCB', 'HCN', 'NH', 'NB', 'NE', 'NG', 'NI', 'NT', 'NS', 'NO', 'RGY', 'AVCA', 'AVCB',
     'AVPH', 'AVPS', 'NHBO', 'ENER', 'DOPE', 'DFIRE', 'GOAP', 'RWDD', 'RWOD', 'PLGSC',
     'PMSUB', 'FRST', 'FRRAP', 'FRSOL', 'FRHYD', 'FRTOR', 'CHVOL', 'ANUA1', 'ANUA2', 'ANUA3', 'ANUA4']

strMethodFeatureRanking = ','.join([str(x) for x in methodFeatureRanking])
strFeaturesName         = ','.join([str(x) for x in featuresName])

pythonFileNamePath = os.path.join( os.path.sep, os.getcwd(), 'ToolPVarAnalysisByMethod_2.py' )
os.system( 'python '+ pythonFileNamePath +' '+ROOT_PATH+' '+SIMULATION_NAME1+' '+
           SIMULATION_NAME2 +' '+DATABASENAME+' '+ strMethodFeatureRanking+ ' '+ strFeaturesName+ ' '+
           str(numberOfFeatures) + ' ' + str(maxNumberOfNeuronsByLayer)+' '+ str(maxNumOfExp)+' '+str(maxNumOfTrials)+ ' '+
           str(trainFactor)+ ' '+ method)