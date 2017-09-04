# Using the magic encoding
# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt; plt.rcdefaults()
import os
import numpy as np
import matplotlib.patches as mpatches
import  pandas as pd
from matplotlib.ticker import MaxNLocator
def saveFeatureImportanceScore ( lstScoreNames, lstScoreValues, pathName, method, numberFeatureToShow, varColor):
    """
    This functions shows the feature importance without variacion in the time on a barchart is showed.     
    :param dfFeaturesScore: This Dataframe has the name of the features and their score, this dataframe is sorted by score and then is send to do the barchart
    :param numberFeatureToShow: In case that the number of features is too hight then this parameter  is used to limit the number of features used in the barchar
    :param varColor: Color of the barchart grafic
    """
    #Tips to grafic
    #plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
    #plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
    #plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    #plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    #plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    #plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
    #plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    plt.rc('ytick', labelsize=6)
    plt.rc('xtick', labelsize=6)
    en_Title     = "Feature Importance"
    en_xLabel    = "Importance"
    pt_Title     = u"Importância dos Features"
    pt_xLabel    = u"Importância"

    #objects      = list(dfFeaturesScore.ix[0:numberFeatureToShow, 0])
    #objects       = lstScoreNames[0:numberFeatureToShow]
    #objects      = list(dfFeaturesScore.ix[dfFeaturesScore.shape[0]-numberFeatureToShow-1:dfFeaturesScore.shape[0],0])
    objects      = list(reversed(lstScoreNames[0:numberFeatureToShow]))
    y_pos        = np.arange(len(objects))
    #performance  = dfFeaturesScore.ix[dfFeaturesScore.shape[0]-numberFeatureToShow-1:dfFeaturesScore.shape[0],1]
    performance  = list(reversed(lstScoreValues[0:numberFeatureToShow]))


    plt.barh(y_pos, performance, align='center', color=varColor)
    plt.yticks(y_pos, objects)
    plt.xlabel(en_xLabel)
    plt.title(en_Title)
    plt.savefig(os.path.join(os.path.sep, pathName, method+'_EPS_'+str(numberFeatureToShow+1)+'_barFeatureImportance_en.eps'), dpi=1200)
    plt.savefig(os.path.join(os.path.sep, pathName, method+'_SVG_'+str(numberFeatureToShow+1)+'_barFeatureImportance_en.svg'), dpi=1200)
    plt.xlabel(pt_xLabel)
    plt.title(pt_Title)
    plt.savefig(os.path.join(os.path.sep, pathName, method+'_EPS_'+str(numberFeatureToShow+1)+'_barFeatureImportance_pt.eps'), dpi=1200)
    plt.savefig(os.path.join(os.path.sep, pathName, method+'_SVG_'+str(numberFeatureToShow+1)+'_barFeatureImportance_pt.svg'), dpi=1200)
    plt.rc('ytick', labelsize=10)
    plt.rc('xtick', labelsize=10)
    plt.cla()
    plt.close()
def saveFeatureImportanceScoreByType ( lstScoreNames, lstScoreValues, pathName,
                                       method, numberFeatureToShow, colorByFeatureType,
                                       featureByType ):
    lstScoreNamesRev = list(reversed(lstScoreNames[0:numberFeatureToShow]))
    lstScoreValuesRev =list(reversed(lstScoreValues[0:numberFeatureToShow]))
    lstFeaturesTypes=[]
    for name in lstScoreNamesRev:
        lstFeaturesTypes.append(featureByType[name])
    d = {'typ': lstFeaturesTypes, 'value': lstScoreValuesRev}
    df = pd.DataFrame(d)
    en_Title = "Feature Importance"
    en_xLabel = "Importance"
    pt_Title = u"Importância dos Features"
    pt_xLabel = u"Importância"
    plt.rc('ytick', labelsize=6)
    plt.rc('xtick', labelsize=6)
    y_pos = np.arange(len(lstScoreNamesRev))



    plt.barh(y_pos, df['value'], align='center',color=[colorByFeatureType[t] for t in df['typ']])


    type_patch_1 = mpatches.Patch(color='#92D050', label='Psychochemical')
    type_patch_2 = mpatches.Patch(color='#EEB5AA', label='Secondary Structure')
    type_patch_3 = mpatches.Patch(color='#FFC000', label='Structural Features')
    type_patch_4 = mpatches.Patch(color='#6699FF', label='Energy')
    plt.legend(handles=[type_patch_1, type_patch_2, type_patch_3, type_patch_4], title='Feature Types', fancybox=True,
               shadow=True)

    plt.yticks(y_pos, lstScoreNamesRev)
    plt.xlabel(en_xLabel)
    plt.title(en_Title)
    plt.savefig(os.path.join(os.path.sep,pathName,method+'_EPS_'+'ByType_'+str(numberFeatureToShow + 1)+'_barFeatureImportance_en.eps'),dpi=1200)
    plt.savefig(os.path.join(os.path.sep,pathName,method+'_SVG_'+'ByType_'+str(numberFeatureToShow + 1)+'_barFeatureImportance_en.svg'),dpi=1200)
    plt.xlabel(pt_xLabel)
    plt.title(pt_Title)


    plt.savefig(os.path.join(os.path.sep,pathName,method+'_EPS_'+'ByType_'+str(numberFeatureToShow + 1)+'_barFeatureImportance_pt.eps'),dpi=1200)
    plt.savefig(os.path.join(os.path.sep,pathName,method+'_SVG_'+'ByType_'+str(numberFeatureToShow + 1)+'_barFeatureImportance_pt.svg'),dpi=1200)
    plt.rc('ytick', labelsize=10)
    plt.rc('xtick', labelsize=10)
    plt.cla()
    plt.close()
def saveFeatureImportanceScoreByTypeG ( lstScoreNames, lstScoreValues, pathName,
                                       method, numberFeatureToShow, colorByFeatureType,
                                       featureByType ):
    lstScoreNamesRev = list(reversed(lstScoreNames[0:numberFeatureToShow]))
    lstScoreValuesRev =list(reversed(lstScoreValues[0:numberFeatureToShow]))
    lstFeaturesTypes=[]
    for name in lstScoreNamesRev:
        lstFeaturesTypes.append(featureByType[name])
    d = {'typ': lstFeaturesTypes, 'value': lstScoreValuesRev}
    df = pd.DataFrame(d)
    en_Title = "Feature Type"
    en_xLabel = "Importance"
    pt_Title = u"Tipo de Feature"
    pt_xLabel = u"Importância"
    plt.rc('ytick', labelsize=6)
    plt.rc('xtick', labelsize=6)
    y_pos = np.arange(len(lstScoreNamesRev))



    plt.barh(y_pos, df['value'], align='center',color=[colorByFeatureType[t] for t in df['typ']])


    #type_patch_1 = mpatches.Patch(color='#92D050', label='Psychochemical')
    #type_patch_2 = mpatches.Patch(color='#EEB5AA', label='Secondary Structure')
    #type_patch_3 = mpatches.Patch(color='#FFC000', label='Structural Features')
    #type_patch_4 = mpatches.Patch(color='#6699FF', label='Energy')
    #plt.legend(handles=[type_patch_1, type_patch_2, type_patch_3, type_patch_4], title='Feature Types', fancybox=True,
    #           shadow=True)

    plt.yticks(y_pos, lstScoreNamesRev)
    plt.xlabel(en_xLabel)
    plt.title(en_Title)
    plt.savefig(os.path.join(os.path.sep,pathName,method+'_EPS_'+'ByType_'+str(numberFeatureToShow + 1)+'_barFeatureImportance_en.eps'),dpi=1200)
    plt.savefig(os.path.join(os.path.sep,pathName,method+'_SVG_'+'ByType_'+str(numberFeatureToShow + 1)+'_barFeatureImportance_en.svg'),dpi=1200)
    plt.xlabel(pt_xLabel)
    plt.title(pt_Title)


    plt.savefig(os.path.join(os.path.sep,pathName,method+'_EPS_'+'ByType_'+str(numberFeatureToShow + 1)+'_barFeatureImportance_pt.eps'),dpi=1200)
    plt.savefig(os.path.join(os.path.sep,pathName,method+'_SVG_'+'ByType_'+str(numberFeatureToShow + 1)+'_barFeatureImportance_pt.svg'),dpi=1200)
    plt.rc('ytick', labelsize=10)
    plt.rc('xtick', labelsize=10)
    plt.cla()
    plt.close()
def saveEvolutionProcess ( lstEvolutionValues, pathName,  type='Fitness',  ):
    if ( type=='Fitness' ):
        plt.plot(lstEvolutionValues, color='red', linewidth=2)
        plt.xlabel('Generations')
        plt.ylabel('Fitness')
        plt.title('Genetic Algorithm Evolution')
        plt.savefig(os.path.join(os.path.sep, pathName, "EPS_EvolutionProcess_en.eps"), dpi=1200)
        plt.xlabel(u'Generações')
        plt.ylabel(u'Aptidão Média')
        plt.title(u'Evolução do Algoritmo Genético')
        plt.savefig(os.path.join(os.path.sep, pathName, "EPS_EvolutionProcess_pt.eps"), dpi=1200)
        plt.cla()
        plt.close()
    else:
        plt.plot(lstEvolutionValues, color='blue', linewidth=2)
        plt.xlabel('Generations')
        plt.ylabel('Average Standard Deviation')
        plt.title('Diversity Analysis')
        plt.savefig(os.path.join(os.path.sep, pathName, "EPS_EvolutionDiversityProcess_en.eps"), dpi=1200)
        plt.xlabel(u'Generações')
        plt.ylabel(u'Desvio Padrão Médio')
        plt.title(u'Analise de Diversidade')
        plt.savefig(os.path.join(os.path.sep, pathName, "EPS_EvolutionDiversityProcess_pt.eps"), dpi=1200)
        plt.cla()
        plt.close()

def saveAnalysisByMethod ( varList, varListValue, typeVarName, method, pathName ):
    plt.rc('ytick', labelsize=6.5)
    plt.rc('xtick', labelsize=6.5)
    ax = plt.figure().gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xlabel('Features Number')
    if typeVarName=="R2":
        plt.ylabel('$R^2$')
        varColor = 'green'
    elif(typeVarName=="RA"):
        plt.ylabel('$R_a$')
        varColor = 'red'
    elif (typeVarName == "RE"):
        plt.ylabel('$R_e$')
        varColor = 'blue'
    elif(typeVarName=='RMSE'):
        plt.ylabel('RMSE')
        varColor = 'magenta'
    elif (typeVarName == 'ERROR'):
        plt.ylabel('Error')
        varColor = 'red'
    plt.title('Number of Features Analysis')
    plt.plot(varList, varListValue, '-b', color=varColor)
    if (typeVarName =='ERROR' or typeVarName=='RMSE'):
        infl_max_index = varListValue.index(min(varListValue))  # get the index of the maximum inflation
    else:
        infl_max_index = varListValue.index(max(varListValue))  # get the index of the maximum inflation
        ax.set_ylim(ymin=0)
        ax.set_ylim(ymax=1)
    ax.set_xlim(xmin=0)
    infl_max = varListValue[infl_max_index]  # get the inflation corresponding to this index
    year_max = varList[infl_max_index]  # get the year corresponding to this index
    plt.plot(year_max, infl_max, 'bo', color='cyan', mfc='none' )
    plt.plot((year_max, year_max), (infl_max, 0),'c--', linewidth=0.9)
    plt.plot((year_max, 0), (infl_max, infl_max), 'c--', linewidth=0.9)
    plt.grid()
    plt.savefig(os.path.join(os.path.sep, pathName,method+"_"+typeVarName+"_EPS_NumVarAnalysis.eps"), dpi=1200)
    plt.rc('ytick', labelsize=10)
    plt.rc('xtick', labelsize=10)
    plt.cla()
    plt.close()

def saveAnalysisByMethod2 ( varList, varR2AdjListValue, varR2ListValue,  method, pathName ):
    plt.rc('ytick', labelsize=6.5)
    plt.rc('xtick', labelsize=6.5)
    ax = plt.figure().gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xlabel('Features Number')
    plt.ylabel('R')
    plt.title('Number of Features Analysis')
    plt.plot(varList, varR2AdjListValue, '-b', color='red')
    plt.plot(varList, varR2ListValue, '-b', color='green')
    #plt.plot(varList, varSpearman, '-b', color='orange')
    #plt.plot(varList, varRMSE, '-b', color='cyan')
    type_patch_1 = mpatches.Patch(color='red',    label='$R_a$')
    type_patch_2 = mpatches.Patch(color='green',  label='$R^2$')
    #type_patch_3 = mpatches.Patch(color='orange', label='R Spearman')
    #type_patch_4 = mpatches.Patch(color='cyan',   label='RMSE')
    plt.legend(handles=[type_patch_1, type_patch_2], title="Methods", fancybox=True,
               shadow=True)

    #plt.xticks( varList)
    infl_max_index = varR2AdjListValue.index(max(varR2AdjListValue))  # get the index of the maximum inflation
    infl_max       = varR2AdjListValue[infl_max_index]  # get the inflation corresponding to this index
    year_max       = varList[infl_max_index]  # get the year corresponding to this index
    plt.plot(year_max, infl_max, 'bo', color='blue', mfc='none' )
    ax.set_ylim(ymin=0)
    ax.set_ylim(ymax=1)
    ax.set_xlim(xmin=0)
    plt.plot((year_max, year_max), (infl_max, 0),'b--', linewidth=0.9)
    plt.plot((year_max,0 ), (infl_max, infl_max ), 'b--', linewidth=0.9)
    plt.grid()
    plt.savefig(os.path.join(os.path.sep, pathName, method+"_EPS_NumVarAnalysis.eps"), dpi=1200)
    plt.rc('ytick', labelsize=10)
    plt.rc('xtick', labelsize=10)
    plt.cla()
    plt.close()