from __future__ import division, absolute_import

import astropy.stats
import glob
import math
import matplotlib.pyplot as plt 
from matplotlib import ticker
from matplotlib.ticker import FormatStrFormatter
import numpy as np 
import os
import pandas as pd
from scipy import integrate,optimize,spatial

three_dex_abd_matched = {'1': ([[ 0.21524025,  0.20638555,  0.18613791,  0.17004973,  0.17328601,
          0.17824797,  0.22330475,  0.27547932,  0.35097406,  0.46594156,
          0.6529005 ,  0.6352904 ,  0.73630952],
        [ 0.01130106,  0.01165314,  0.00956257,  0.0100338 ,  0.00851037,
          0.00829444,  0.00939837,  0.0112802 ,  0.01710905,  0.01586881,
          0.01895867,  0.05304972,  0.06702147]]),
 '10': ([[ 0.0434983 ,  0.04448739,  0.03900445,  0.03950445,  0.03455255,
          0.03613353,  0.03198477,  0.02779419,  0.04298508,  0.05409842,
          0.08323442,  0.13483586,  0.1875    ],
        [ 0.00907987,  0.01003662,  0.00746137,  0.00781005,  0.00876654,
          0.00778842,  0.00710085,  0.00787262,  0.00931001,  0.01068675,
          0.01922668,  0.03250244,  0.05831473]]),
 '2': ([[ 0.12064069,  0.12118292,  0.11193204,  0.10343818,  0.09727599,
          0.09318955,  0.09715361,  0.12339972,  0.16939451,  0.2670205 ,
          0.41188724,  0.50939394,  0.52678571],
        [ 0.01189233,  0.01312654,  0.01054764,  0.01001456,  0.00993245,
          0.01071466,  0.00966967,  0.0091511 ,  0.01455119,  0.01775133,
          0.01885678,  0.05781321,  0.05745782]]),
 '20': ([[ 0.02384301,  0.02535176,  0.02022905,  0.02055364,  0.01931314,
          0.017112  ,  0.01431874,  0.01258658,  0.0159481 ,  0.01943668,
          0.03090288,  0.07816919,  0.07916667],
        [ 0.00626565,  0.00685997,  0.00526008,  0.00613409,  0.00587699,
          0.00553958,  0.00485825,  0.00502655,  0.00554471,  0.0062414 ,
          0.01229515,  0.01883964,  0.02942593]]),
 '3': ([[ 0.09119876,  0.09777328,  0.08925934,  0.08259272,  0.07711375,
          0.07068848,  0.065675  ,  0.07988132,  0.11103945,  0.1773137 ,
          0.28889296,  0.40550505,  0.37321429],
        [ 0.01140526,  0.0122457 ,  0.00999486,  0.00918654,  0.00988064,
          0.00946682,  0.00857933,  0.00827363,  0.01327769,  0.01686665,
          0.01616292,  0.0496007 ,  0.06285859]]),
 '5': ([[ 0.06871318,  0.06882041,  0.06418409,  0.05834665,  0.05640096,
          0.05266543,  0.04813125,  0.05107188,  0.07282119,  0.09718295,
          0.19193237,  0.26286616,  0.28333333],
        [ 0.01079781,  0.01208153,  0.01074775,  0.00901779,  0.00918551,
          0.0088578 ,  0.00872636,  0.00841686,  0.01211816,  0.01476584,
          0.01849396,  0.0521784 ,  0.06123724]])}

two_dex_abd_matched = {'1': ([[ 0.2277589 ,  0.20929902,  0.20440717,  0.19045579,  0.17335294,
          0.17895473,  0.21237887,  0.2784413 ,  0.40348453,  0.5576678 ,
          0.72205984,  0.87900008,  0.92788462],
        [ 0.01441421,  0.01248772,  0.01319243,  0.01204725,  0.01152511,
          0.00900664,  0.0116242 ,  0.01143036,  0.01337072,  0.01665733,
          0.02369187,  0.01932513,  0.05786732]]),
 '10': ([[ 0.04756858,  0.04980623,  0.05146618,  0.04804556,  0.04270754,
          0.03740174,  0.03831069,  0.03824497,  0.04121288,  0.06389169,
          0.12307228,  0.2865359 ,  0.43269231],
        [ 0.00942242,  0.01033441,  0.01064282,  0.00874843,  0.00855415,
          0.00720339,  0.00656024,  0.00555056,  0.00648568,  0.01294673,
          0.02002825,  0.04360923,  0.11830788]]),
 '2': ([[ 0.12990243,  0.12738903,  0.12694609,  0.11571515,  0.10560929,
          0.09257186,  0.09617228,  0.11780891,  0.1859156 ,  0.33015289,
          0.51666957,  0.7644459 ,  0.87660256],
        [ 0.01349072,  0.01281358,  0.01434831,  0.0103789 ,  0.01165435,
          0.00854937,  0.00869274,  0.00846095,  0.00808849,  0.02341006,
          0.0219697 ,  0.02357977,  0.06410006]]),
 '20': ([[ 0.02597944,  0.02514224,  0.02744873,  0.02401747,  0.02247834,
          0.02085579,  0.02042847,  0.02035271,  0.02037985,  0.02690943,
          0.06062737,  0.10727761,  0.15224359],
        [ 0.00689687,  0.00717086,  0.00797024,  0.00599567,  0.006135  ,
          0.00550897,  0.00578622,  0.00404915,  0.00486626,  0.00822912,
          0.01503631,  0.03598   ,  0.05570783]]),
 '3': ([[ 0.10125596,  0.10313103,  0.10249001,  0.09152116,  0.08442258,
          0.07670431,  0.07217208,  0.08102919,  0.11020136,  0.23244375,
          0.39091166,  0.63650154,  0.80528846],
        [ 0.01285048,  0.01269348,  0.0142144 ,  0.01074026,  0.01016215,
          0.00845994,  0.00754706,  0.00660626,  0.00947444,  0.01941087,
          0.02452525,  0.03119116,  0.0615887 ]]),
 '5': ([[ 0.07707357,  0.07497129,  0.07521926,  0.06941231,  0.06047828,
          0.05585792,  0.05540505,  0.05540863,  0.06323059,  0.12723545,
          0.24226817,  0.48849221,  0.64983974],
        [ 0.01257445,  0.01240699,  0.01306162,  0.0089147 ,  0.01045907,
          0.00876058,  0.00736706,  0.00684428,  0.00782831,  0.01641762,
          0.02766064,  0.02216969,  0.11269876]])}

one_dex_abd_matched = {'1': ([[ 0.24090065,  0.21751226,  0.21489993,  
            0.1894229 ,  0.18796087,
          0.17726431,  0.20180639,  0.27350405,  0.42899167,  0.66968654,
          0.87864981,  0.95474644,  1.        ],
        [ 0.01145703,  0.01261648,  0.01349425,  0.01143488,  0.0098266 ,
          0.00890501,  0.00808996,  0.00900656,  0.01142553,  0.01202355,
          0.01441284,  0.01614769,  0.        ]]),
 '10': ([[ 0.05282593,  0.05287924,  0.05178348,  0.04819594,  0.04430741,
          0.03937763,  0.03436343,  0.03209625,  0.03886541,  0.06173756,
          0.12849462,  0.43421778,  0.7390625 ],
        [ 0.0107828 ,  0.01097779,  0.01009919,  0.00905892,  0.00782053,
          0.00749636,  0.00665324,  0.00744886,  0.00935906,  0.01220939,
          0.02361977,  0.04500612,  0.0672377 ]]),
 '2': ([[ 0.13947179,  0.13403766,  0.13205966,  0.11856716,  0.11513654,
          0.09391647,  0.09153797,  0.09952045,  0.1774149 ,  0.39959388,
          0.72658597,  0.92456062,  0.9921875 ],
        [ 0.01208771,  0.01360823,  0.01351236,  0.0117967 ,  0.00948068,
          0.00906088,  0.00822835,  0.00907392,  0.013738  ,  0.01755909,
          0.01341753,  0.02104195,  0.00730792]]),
 '20': ([[ 0.02637069,  0.02678547,  0.02904849,  0.02689762,  0.0232155 ,
          0.02239701,  0.01759682,  0.0208054 ,  0.02202997,  0.03431871,
          0.04782364,  0.18408795,  0.38229167],
        [ 0.0077015 ,  0.00753392,  0.00748456,  0.00673008,  0.00569602,
          0.00594712,  0.0051561 ,  0.0057339 ,  0.00707003,  0.00808574,
          0.0113854 ,  0.03388804,  0.06644556]]),
 '3': ([[ 0.10697682,  0.10539415,  0.10845122,  0.09131209,  0.08983389,
          0.07622917,  0.06914106,  0.06562035,  0.08964582,  0.24107919,
          0.53426499,  0.85987446,  0.9921875 ],
        [ 0.01349656,  0.01286965,  0.01258807,  0.01156779,  0.01004648,
          0.00974959,  0.00908036,  0.00846809,  0.01190788,  0.0186402 ,
          0.02561799,  0.0220085 ,  0.00730792]]),
 '5': ([[ 0.0829296 ,  0.0767095 ,  0.08028205,  0.06873084,  0.0658115 ,
          0.05865908,  0.05155796,  0.04898616,  0.05256302,  0.10856876,
          0.31840299,  0.71632312,  0.9296875 ],
        [ 0.01293653,  0.0132994 ,  0.01202746,  0.01024547,  0.00966556,
          0.01024098,  0.00791901,  0.00852191,  0.01026344,  0.01356835,
          0.02565866,  0.03820398,  0.04016458]])}

one_dex_norm = {'1': ([[ 0.23379138,  0.21858028,  0.21544219,  0.19484084,  0.19169834,
          0.17881306,  0.18925336,  0.2509041 ,  0.42610801,  0.6986764 ,
          0.91731741,  0.9875    ,  1.        ],
        [ 0.01093244,  0.01359421,  0.01257943,  0.01314939,  0.00962991,
          0.00968802,  0.00884824,  0.00916126,  0.00944932,  0.00868739,
          0.01068788,  0.01169268,  0.        ]]),
 '10': ([[ 0.05097433,  0.05342309,  0.05144121,  0.04976471,  0.04664067,
          0.03953891,  0.03558171,  0.03403173,  0.03652341,  0.07052831,
          0.1808226 ,  0.65861222,  0.93333333],
        [ 0.01032246,  0.01117695,  0.00979262,  0.00938893,  0.00863373,
          0.00705196,  0.00760729,  0.00725605,  0.00863966,  0.01411   ,
          0.02977818,  0.06330736,  0.05270463]]),
 '2': ([[ 0.13506372,  0.13455398,  0.13282997,  0.12184482,  0.11706111,
          0.09973883,  0.0930339 ,  0.09587241,  0.17229977,  0.42859355,
          0.8063813 ,  0.97329545,  1.        ],
        [ 0.01143441,  0.01415717,  0.01295881,  0.01255686,  0.01064479,
          0.00920652,  0.00925522,  0.00774904,  0.01212893,  0.01616638,
          0.01843806,  0.01429375,  0.        ]]),
 '20': ([[ 0.02543328,  0.0270065 ,  0.02892489,  0.02787714,  0.02374327,
          0.02240617,  0.01976756,  0.02039297,  0.02114825,  0.03430814,
          0.07321814,  0.33127289,  0.71111111],
        [ 0.00739287,  0.00762563,  0.00721275,  0.00680826,  0.00619374,
          0.00590446,  0.00584163,  0.00568387,  0.00669582,  0.00874335,
          0.01914046,  0.04155249,  0.12668616]]),
 '3': ([[ 0.10356548,  0.10577668,  0.10848871,  0.09457448,  0.09156254,
          0.08007504,  0.07148496,  0.06546403,  0.08749008,  0.27215413,
          0.63875407,  0.92082293,  1.        ],
        [ 0.01284947,  0.01325919,  0.0120804 ,  0.01213016,  0.01076404,
          0.00973611,  0.0104359 ,  0.00791612,  0.01108596,  0.01368561,
          0.02065483,  0.01528594,  0.        ]]),
 '5': ([[ 0.08011167,  0.07727852,  0.08043464,  0.07125348,  0.06703718,
          0.05997341,  0.05343608,  0.04997222,  0.05174344,  0.12648361,
          0.43564471,  0.86068307,  1.        ],
        [ 0.01236788,  0.01359278,  0.01161002,  0.01091639,  0.01023869,
          0.010021  ,  0.00971142,  0.00798677,  0.0098478 ,  0.01536732,
          0.02288673,  0.0229637 ,  0.        ]])}

two_dex_norm = {'1': ([[ 0.21757995,  0.21159866,  0.20698244,  0.19267024,  0.17728934,
          0.17941772,  0.19704583,  0.25948005,  0.35904919,  0.54708736,
          0.71595682,  0.92759048,  0.96875   ],
        [ 0.01413842,  0.01217976,  0.01275151,  0.01224816,  0.01500227,
          0.00765373,  0.00840479,  0.00696645,  0.01410345,  0.01224488,
          0.016043  ,  0.02031895,  0.0292317 ]]),
 '10': ([[ 0.04526536,  0.05056117,  0.05125827,  0.04947443,  0.04393309,
          0.03918201,  0.03899048,  0.03712702,  0.03831807,  0.06205462,
          0.11817761,  0.2671782 ,  0.578125  ],
        [ 0.00921302,  0.01045191,  0.01023102,  0.00867248,  0.0101528 ,
          0.00769676,  0.00646166,  0.00581073,  0.00709351,  0.00922964,
          0.02032159,  0.03135078,  0.10820242]]),
 '2': ([[ 0.123864  ,  0.12859161,  0.12776939,  0.11990065,  0.10803898,
          0.09396226,  0.09306344,  0.10582313,  0.16855359,  0.31243535,
          0.49743619,  0.81120753,  0.93125   ],
        [ 0.01326967,  0.01305198,  0.01395815,  0.0101661 ,  0.01439067,
          0.0086511 ,  0.00886102,  0.00516303,  0.01236694,  0.01541778,
          0.03050539,  0.024729  ,  0.03743484]]),
 '20': ([[ 0.0247306 ,  0.02549198,  0.02712836,  0.02517137,  0.02344174,
          0.02226584,  0.01960278,  0.02063911,  0.01744428,  0.03040506,
          0.04858382,  0.12446845,  0.23125   ],
        [ 0.00672988,  0.0072824 ,  0.00776997,  0.00592793,  0.00697547,
          0.00588343,  0.00562093,  0.00458423,  0.00401182,  0.00831327,
          0.0099017 ,  0.03118885,  0.11204334]]),
 '3': ([[ 0.09632808,  0.10458555,  0.10365274,  0.09462682,  0.08569358,
          0.07928289,  0.07294874,  0.07407289,  0.10272467,  0.21203764,
          0.369383  ,  0.63161268,  0.909375  ],
        [ 0.01259439,  0.01283883,  0.01378067,  0.01041436,  0.01234633,
          0.00876314,  0.00798071,  0.00501344,  0.01139153,  0.01256465,
          0.03044705,  0.03757055,  0.04799445]]),
 '5': ([[ 0.07351418,  0.07606423,  0.07561372,  0.0727972 ,  0.06071745,
          0.05857334,  0.05507502,  0.05394583,  0.06087093,  0.11975757,
          0.22611026,  0.43858088,  0.759375  ],
        [ 0.01234074,  0.01247395,  0.01269944,  0.00886458,  0.01191216,
          0.00933791,  0.00800756,  0.00547326,  0.00879668,  0.01124001,
          0.03046099,  0.0325339 ,  0.07499186]])}

three_dex_norm = {'1': ([[ 0.20829433,  0.20457921,  0.1903796 ,  0.17148667,  0.17655332,
          0.17448569,  0.1998673 ,  0.24076067,  0.32940725,  0.40774115,
          0.5455842 ,  0.71231361,  0.69908494],
        [ 0.01089732,  0.01200659,  0.01101784,  0.00935578,  0.00896163,
          0.00846024,  0.00860821,  0.00934991,  0.00732583,  0.014189  ,
          0.01834871,  0.01754357,  0.04036407]]),
 '10': ([[ 0.04203558,  0.04419473,  0.03941548,  0.04026057,  0.03451402,
          0.032655  ,  0.03592204,  0.030436  ,  0.03527852,  0.04144363,
          0.07733356,  0.09325203,  0.10696836],
        [ 0.00882999,  0.01021811,  0.00706577,  0.0075648 ,  0.00862625,
          0.00769305,  0.00803968,  0.0082279 ,  0.00623344,  0.00972556,
          0.01211075,  0.01743087,  0.0281395 ]]),
 '2': ([[ 0.11636379,  0.12017516,  0.11518941,  0.10379404,  0.10274423,
          0.09201899,  0.09275208,  0.10696531,  0.1555338 ,  0.19832555,
          0.33570517,  0.45564685,  0.53448623],
        [ 0.01180342,  0.0129437 ,  0.01054841,  0.01041868,  0.00938872,
          0.01039477,  0.01126146,  0.00913982,  0.00836321,  0.01602994,
          0.01959266,  0.02334625,  0.04419489]]),
 '20': ([[ 0.02315809,  0.02489594,  0.02076118,  0.02113088,  0.01889123,
          0.01773325,  0.01611588,  0.0146317 ,  0.01397966,  0.01363161,
          0.03079506,  0.04637692,  0.05099439],
        [ 0.00602487,  0.00689501,  0.0054539 ,  0.00608519,  0.00594935,
          0.00571235,  0.00517594,  0.00510871,  0.00529972,  0.00529514,
          0.00916591,  0.01038736,  0.0146574 ]]),
 '3': ([[ 0.0879212 ,  0.09649538,  0.09177762,  0.08309075,  0.08026366,
          0.07006718,  0.06937523,  0.06802135,  0.09895054,  0.13284495,
          0.22807118,  0.32004935,  0.39108318],
        [ 0.01123558,  0.01218644,  0.00999191,  0.00972049,  0.00925183,
          0.00936082,  0.01052106,  0.01015136,  0.00779289,  0.01288677,
          0.0203126 ,  0.036634  ,  0.03673733]]),
 '5': ([[ 0.06622738,  0.0680997 ,  0.06545554,  0.05953069,  0.057135  ,
          0.05071328,  0.05087791,  0.05021138,  0.06142949,  0.07497064,
          0.1399526 ,  0.20886604,  0.27510545],
        [ 0.01063068,  0.01204707,  0.01087383,  0.0092756 ,  0.00945984,
          0.00807972,  0.01016534,  0.01027421,  0.00523548,  0.01343365,
          0.01911088,  0.02543038,  0.04120676]])}

bin_centers = np.array([  9.2,   9.4,   9.6,   9.8,  10. ,  10.2,  10.4,  10.6,  10.8,
        11. ,  11.2,  11.4,  11.6])

def plot_mean_halo_frac(bin_centers,mean_vals,ax,std,plot_idx,color='grey',\
  label=None,text=False):
    ax.errorbar(bin_centers,mean_vals,yerr=std,color=color,label=label)
    if text == True:
      titles = [1,2,3,5,10,20]
      title_here = 'n = {0}'.format(titles[plot_idx])
      ax.text(0.05, 0.3, title_here,horizontalalignment='left',            \
            verticalalignment='bottom',transform=ax.transAxes,fontsize=18)
    if plot_idx == 0:
        ax.legend(loc='best')

neigh_vals = np.array([1,2,3,5,10,20])

nrow = int(2)
ncol = int(3)

fig,axes = plt.subplots(nrows=nrow,ncols=ncol,                        \
    figsize=(100,200),sharex=True,sharey=True)
axes_flat = axes.flatten()

zz = int(0)
while zz <=4:
    for jj in neigh_vals:
        nn_str = '{0}'.format(jj)
        plot_mean_halo_frac(bin_centers,three_dex_abd_matched[nn_str][0],\
            axes_flat[zz],three_dex_abd_matched[nn_str][1],zz,\
            color='darkmagenta',label='0.3 ABD')    
        plot_mean_halo_frac(bin_centers,two_dex_abd_matched[nn_str][0],\
            axes_flat[zz],two_dex_abd_matched[nn_str][1],zz,\
            color='lime',label='0.2 ABD')
        plot_mean_halo_frac(bin_centers,one_dex_abd_matched[nn_str][0],\
            axes_flat[zz],one_dex_abd_matched[nn_str][1],zz,\
            color='red',label='0.1 ABD')
        plot_mean_halo_frac(bin_centers,three_dex_norm[nn_str][0],\
            axes_flat[zz],three_dex_abd_matched[nn_str][1],zz,\
            color='indigo',label='0.3')    
        plot_mean_halo_frac(bin_centers,two_dex_abd_matched[nn_str][0],\
            axes_flat[zz],two_dex_norm[nn_str][1],zz,\
            color='seagreen',label='0.2')
        plot_mean_halo_frac(bin_centers,one_dex_abd_matched[nn_str][0],\
            axes_flat[zz],one_dex_norm[nn_str][1],zz,\
            color='maroon',label='0.1',text=True)
        zz += 1

plt.subplots_adjust(top=0.97,bottom=0.1,left=0.03,right=0.99,hspace=0.10,\
    wspace=0.12)     

plt.show()          