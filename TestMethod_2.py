import os

ROOT_PATH        = '/home/edwin/SIMULATIONS'
SIMULATION_NAME1 = '4STATE'
SIMULATION_NAME2 = '1hdd-C'
DATABASENAME     = '1hdd-C_protein_features.csv'

numberOfFeatures          = 37
maxNumberOfNeuronsByLayer = 10
maxNumOfExp               = 10
maxNumOfTrials            = 30
trainFactor               = 0.9
method                    ="M2"



pythonFileNamePath      = os.path.join( os.path.sep, os.getcwd(), 'ToolPMethod2.py' )
os.system( 'python '+ pythonFileNamePath +' '+ROOT_PATH+' '+SIMULATION_NAME1+' '+
           SIMULATION_NAME2 +' '+DATABASENAME+' '+ str(numberOfFeatures)+ ' '+
           str(maxNumOfExp) +' '+str(maxNumOfTrials)+' '+str(trainFactor)+' '+ method )


