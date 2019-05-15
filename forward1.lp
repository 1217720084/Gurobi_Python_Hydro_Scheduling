\ Model ForwardSolve
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize multi-objectives
  OBJ0: Priority=0 Weight=1 AbsTol=0 RelTol=0
  
  OBJ1: Priority=1 Weight=1 AbsTol=0 RelTol=0
   2090 generation[1,peak,Huntly_U6] + 1421.2 generation[1,peak,Huntly_U5]
   + 1387.76 generation[1,peak,TCC] + 2691.92 generation[1,peak,NI_Thermal]
   + 5185 generation[1,peak,SI_Thermal]
   + 6270 generation[1,shoulder,Huntly_U6]
   + 4263.6 generation[1,shoulder,Huntly_U5]
   + 4163.28 generation[1,shoulder,TCC]
   + 8075.76 generation[1,shoulder,NI_Thermal]
   + 15555 generation[1,shoulder,SI_Thermal]
   + 9196 generation[1,offpeak,Huntly_U6]
   + 6253.28 generation[1,offpeak,Huntly_U5]
   + 6106.144 generation[1,offpeak,TCC]
   + 11844.448 generation[1,offpeak,NI_Thermal]
   + 22814 generation[1,offpeak,SI_Thermal] + 10600 shortage[1,peak,HY,ls1]
   + 14800 shortage[1,peak,HY,ls2] + 63600 shortage[1,peak,HY,ls3]
   + 105800 shortage[1,peak,HY,ls4] + 211600 shortage[1,peak,HY,ls5]
   + 10600 shortage[1,peak,NI,ls1] + 14800 shortage[1,peak,NI,ls2]
   + 63600 shortage[1,peak,NI,ls3] + 105800 shortage[1,peak,NI,ls4]
   + 211600 shortage[1,peak,NI,ls5] + 10600 shortage[1,peak,SI,ls1]
   + 14800 shortage[1,peak,SI,ls2] + 63600 shortage[1,peak,SI,ls3]
   + 105800 shortage[1,peak,SI,ls4] + 211600 shortage[1,peak,SI,ls5]
   + 31800 shortage[1,shoulder,HY,ls1] + 44400 shortage[1,shoulder,HY,ls2]
   + 190800 shortage[1,shoulder,HY,ls3]
   + 317400 shortage[1,shoulder,HY,ls4]
   + 634800 shortage[1,shoulder,HY,ls5] + 31800 shortage[1,shoulder,NI,ls1]
   + 44400 shortage[1,shoulder,NI,ls2] + 190800 shortage[1,shoulder,NI,ls3]
   + 317400 shortage[1,shoulder,NI,ls4]
   + 634800 shortage[1,shoulder,NI,ls5] + 31800 shortage[1,shoulder,SI,ls1]
   + 44400 shortage[1,shoulder,SI,ls2] + 190800 shortage[1,shoulder,SI,ls3]
   + 317400 shortage[1,shoulder,SI,ls4]
   + 634800 shortage[1,shoulder,SI,ls5] + 46640 shortage[1,offpeak,HY,ls1]
   + 65120 shortage[1,offpeak,HY,ls2] + 279840 shortage[1,offpeak,HY,ls3]
   + 465520 shortage[1,offpeak,HY,ls4] + 931040 shortage[1,offpeak,HY,ls5]
   + 46640 shortage[1,offpeak,NI,ls1] + 65120 shortage[1,offpeak,NI,ls2]
   + 279840 shortage[1,offpeak,NI,ls3] + 465520 shortage[1,offpeak,NI,ls4]
   + 931040 shortage[1,offpeak,NI,ls5] + 46640 shortage[1,offpeak,SI,ls1]
   + 65120 shortage[1,offpeak,SI,ls2] + 279840 shortage[1,offpeak,SI,ls3]
   + 465520 shortage[1,offpeak,SI,ls4] + 931040 shortage[1,offpeak,SI,ls5]
   + futurecost[1] + 10000 arcflowdeficit[1,peak,Lake_Hawea,Lake_Dunstan]
   + 10000 arcflowdeficit[1,peak,Lake_Tekapo,Lake_Scott]
   + 10000 arcflowdeficit[1,peak,Lake_Scott,Tekapo_canal_head]
   + 10000 arcflowdeficit[1,peak,Lake_Scott,Lake_Benmore]
   + 10000 arcflowdeficit[1,peak,Tekapo_canal_head,Tekapo_canal_tail]
   + 10000 arcflowdeficit[1,peak,Lake_Pukaki,Lake_Benmore]
   + 10000 arcflowdeficit[1,peak,Lake_Pukaki,Pukaki_Ohau_canal_junction]
   + 10000 arcflowdeficit[1,peak,Lake_Ohau,Pukaki_Ohau_canal_junction]
   + 10000 arcflowdeficit[1,peak,Lake_Ruataniwha,Lake_Benmore]
   + 30000 arcflowdeficit[1,shoulder,Lake_Hawea,Lake_Dunstan]
   + 30000 arcflowdeficit[1,shoulder,Lake_Tekapo,Lake_Scott]
   + 30000 arcflowdeficit[1,shoulder,Lake_Scott,Tekapo_canal_head]
   + 30000 arcflowdeficit[1,shoulder,Lake_Scott,Lake_Benmore]
   + 30000 arcflowdeficit[1,shoulder,Tekapo_canal_head,Tekapo_canal_tail]
   + 30000 arcflowdeficit[1,shoulder,Lake_Pukaki,Lake_Benmore]
   + 30000 arcflowdeficit[1,shoulder,Lake_Pukaki,Pukaki_Ohau_canal_junction]
   + 30000 arcflowdeficit[1,shoulder,Lake_Ohau,Pukaki_Ohau_canal_junction]
   + 30000 arcflowdeficit[1,shoulder,Lake_Ruataniwha,Lake_Benmore]
   + 44000 arcflowdeficit[1,offpeak,Lake_Hawea,Lake_Dunstan]
   + 44000 arcflowdeficit[1,offpeak,Lake_Tekapo,Lake_Scott]
   + 44000 arcflowdeficit[1,offpeak,Lake_Scott,Tekapo_canal_head]
   + 44000 arcflowdeficit[1,offpeak,Lake_Scott,Lake_Benmore]
   + 44000 arcflowdeficit[1,offpeak,Tekapo_canal_head,Tekapo_canal_tail]
   + 44000 arcflowdeficit[1,offpeak,Lake_Pukaki,Lake_Benmore]
   + 44000 arcflowdeficit[1,offpeak,Lake_Pukaki,Pukaki_Ohau_canal_junction]
   + 44000 arcflowdeficit[1,offpeak,Lake_Ohau,Pukaki_Ohau_canal_junction]
   + 44000 arcflowdeficit[1,offpeak,Lake_Ruataniwha,Lake_Benmore]
   + 10000 arcflowsurplus[1,peak,Lake_Hawea,Lake_Dunstan]
   + 10000 arcflowsurplus[1,peak,Clyde_tail,Lake_Roxburgh]
   + 10000 arcflowsurplus[1,peak,Roxburgh_tail,SEA]
   + 10000 arcflowsurplus[1,peak,Lake_Taupo,Lake_Aratiatia]
   + 10000 arcflowsurplus[1,peak,Karapiro_tail,SEA]
   + 10000 arcflowsurplus[1,peak,Lake_Ohau,Lake_Ruataniwha]
   + 10000 arcflowsurplus[1,peak,Waitaki_tail,SEA]
   + 30000 arcflowsurplus[1,shoulder,Lake_Hawea,Lake_Dunstan]
   + 30000 arcflowsurplus[1,shoulder,Clyde_tail,Lake_Roxburgh]
   + 30000 arcflowsurplus[1,shoulder,Roxburgh_tail,SEA]
   + 30000 arcflowsurplus[1,shoulder,Lake_Taupo,Lake_Aratiatia]
   + 30000 arcflowsurplus[1,shoulder,Karapiro_tail,SEA]
   + 30000 arcflowsurplus[1,shoulder,Lake_Ohau,Lake_Ruataniwha]
   + 30000 arcflowsurplus[1,shoulder,Waitaki_tail,SEA]
   + 44000 arcflowsurplus[1,offpeak,Lake_Hawea,Lake_Dunstan]
   + 44000 arcflowsurplus[1,offpeak,Clyde_tail,Lake_Roxburgh]
   + 44000 arcflowsurplus[1,offpeak,Roxburgh_tail,SEA]
   + 44000 arcflowsurplus[1,offpeak,Lake_Taupo,Lake_Aratiatia]
   + 44000 arcflowsurplus[1,offpeak,Karapiro_tail,SEA]
   + 44000 arcflowsurplus[1,offpeak,Lake_Ohau,Lake_Ruataniwha]
   + 44000 arcflowsurplus[1,offpeak,Waitaki_tail,SEA]
Subject To
 endstoragegwh: - 3.58333e-07 endstorage[1,Lake_Benmore]
   - 2.55472e-07 endstorage[1,Lake_Hawea]
   - 7.28861e-07 endstorage[1,Lake_Ohau]
   - 7.28861e-07 endstorage[1,Lake_Pukaki]
   - 6.86389e-07 endstorage[1,Lake_Taupo]
   - 1.15022e-06 endstorage[1,Lake_Tekapo]
   - 4.25389e-07 endstorage[1,Lakes_Manapouri_Te_Anau]
   - 9.81944e-07 endstorage[1,Lake_Waikaremoana] + endstorage_gwh[vs1]
   + endstorage_gwh[vs2] + endstorage_gwh[vs3] + endstorage_gwh[vs4]
   + endstorage_gwh[vs5] + endstorage_gwh[vs6] + endstorage_gwh[vs7]
   + endstorage_gwh[vs8] + endstorage_gwh[vs9] + endstorage_gwh[vs10]
   + endstorage_gwh[vs11] + endstorage_gwh[vs12] + endstorage_gwh[vs13]
   + endstorage_gwh[vs14] + endstorage_gwh[vs15] + endstorage_gwh[vs16]
   = 0
 futurecost[1]: futurecost[1] + 518440 endstorage_gwh[vs1]
   + 379930 endstorage_gwh[vs2] + 331430 endstorage_gwh[vs3]
   + 251860 endstorage_gwh[vs4] + 204880 endstorage_gwh[vs5]
   + 153090 endstorage_gwh[vs6] + 142890 endstorage_gwh[vs7]
   + 121850 endstorage_gwh[vs8] + 109730 endstorage_gwh[vs9]
   + 87480 endstorage_gwh[vs10] + 77800 endstorage_gwh[vs11]
   + 50000 endstorage_gwh[vs12] + 46140 endstorage_gwh[vs13]
   + 37020 endstorage_gwh[vs14] + 20750 endstorage_gwh[vs15]
   >= 1.0499489800000001e+09
 supplydemandbalance[1,peak,HY]: shortage[1,peak,HY,ls1]
   + shortage[1,peak,HY,ls2] + shortage[1,peak,HY,ls3]
   + shortage[1,peak,HY,ls4] + shortage[1,peak,HY,ls5]
   + powerflow[1,peak,NI,HY] - powerflow[1,peak,HY,NI]
   - powerflow[1,peak,HY,SI] + powerflow[1,peak,SI,HY] = 218.05
 supplydemandbalance[1,peak,NI]: generation[1,peak,Huntly_U6]
   + generation[1,peak,Huntly_U5] + generation[1,peak,TCC]
   + generation[1,peak,NI_Thermal] + generation[1,peak,Arapuni]
   + generation[1,peak,Aratiatia] + generation[1,peak,Atiamuri]
   + generation[1,peak,Karapiro] + generation[1,peak,Maraetai]
   + generation[1,peak,Ohakuri] + generation[1,peak,Waipapa]
   + generation[1,peak,Whakamaru] + generation[1,peak,Mangahao]
   + generation[1,peak,Matahina] + generation[1,peak,Rangipo]
   + generation[1,peak,Tokaanu] + generation[1,peak,Waikaremoana]
   + shortage[1,peak,NI,ls1] + shortage[1,peak,NI,ls2]
   + shortage[1,peak,NI,ls3] + shortage[1,peak,NI,ls4]
   + shortage[1,peak,NI,ls5] - powerflow[1,peak,NI,HY]
   + powerflow[1,peak,HY,NI] = 1352.9
 supplydemandbalance[1,peak,SI]: generation[1,peak,SI_Thermal]
   + generation[1,peak,Aviemore] + generation[1,peak,Benmore]
   + generation[1,peak,Ohau_A] + generation[1,peak,Ohau_B]
   + generation[1,peak,Ohau_C] + generation[1,peak,Tekapo_B]
   + generation[1,peak,Tekapo_A] + generation[1,peak,Waitaki]
   + generation[1,peak,Clyde_220kV] + generation[1,peak,Roxburgh]
   + generation[1,peak,Manapouri] + generation[1,peak,Cobb]
   + generation[1,peak,Coleridge] + shortage[1,peak,SI,ls1]
   + shortage[1,peak,SI,ls2] + shortage[1,peak,SI,ls3]
   + shortage[1,peak,SI,ls4] + shortage[1,peak,SI,ls5]
   + powerflow[1,peak,HY,SI] - powerflow[1,peak,SI,HY] = 1701.9
 supplydemandbalance[1,shoulder,HY]: shortage[1,shoulder,HY,ls1]
   + shortage[1,shoulder,HY,ls2] + shortage[1,shoulder,HY,ls3]
   + shortage[1,shoulder,HY,ls4] + shortage[1,shoulder,HY,ls5]
   + powerflow[1,shoulder,NI,HY] - powerflow[1,shoulder,HY,NI]
   - powerflow[1,shoulder,HY,SI] + powerflow[1,shoulder,SI,HY] = 229.683
 supplydemandbalance[1,shoulder,NI]: generation[1,shoulder,Huntly_U6]
   + generation[1,shoulder,Huntly_U5] + generation[1,shoulder,TCC]
   + generation[1,shoulder,NI_Thermal] + generation[1,shoulder,Arapuni]
   + generation[1,shoulder,Aratiatia] + generation[1,shoulder,Atiamuri]
   + generation[1,shoulder,Karapiro] + generation[1,shoulder,Maraetai]
   + generation[1,shoulder,Ohakuri] + generation[1,shoulder,Waipapa]
   + generation[1,shoulder,Whakamaru] + generation[1,shoulder,Mangahao]
   + generation[1,shoulder,Matahina] + generation[1,shoulder,Rangipo]
   + generation[1,shoulder,Tokaanu] + generation[1,shoulder,Waikaremoana]
   + shortage[1,shoulder,NI,ls1] + shortage[1,shoulder,NI,ls2]
   + shortage[1,shoulder,NI,ls3] + shortage[1,shoulder,NI,ls4]
   + shortage[1,shoulder,NI,ls5] - powerflow[1,shoulder,NI,HY]
   + powerflow[1,shoulder,HY,NI] = 1283.667
 supplydemandbalance[1,shoulder,SI]: generation[1,shoulder,SI_Thermal]
   + generation[1,shoulder,Aviemore] + generation[1,shoulder,Benmore]
   + generation[1,shoulder,Ohau_A] + generation[1,shoulder,Ohau_B]
   + generation[1,shoulder,Ohau_C] + generation[1,shoulder,Tekapo_B]
   + generation[1,shoulder,Tekapo_A] + generation[1,shoulder,Waitaki]
   + generation[1,shoulder,Clyde_220kV] + generation[1,shoulder,Roxburgh]
   + generation[1,shoulder,Manapouri] + generation[1,shoulder,Cobb]
   + generation[1,shoulder,Coleridge] + shortage[1,shoulder,SI,ls1]
   + shortage[1,shoulder,SI,ls2] + shortage[1,shoulder,SI,ls3]
   + shortage[1,shoulder,SI,ls4] + shortage[1,shoulder,SI,ls5]
   + powerflow[1,shoulder,HY,SI] - powerflow[1,shoulder,SI,HY] = 1649.1
 supplydemandbalance[1,offpeak,HY]: shortage[1,offpeak,HY,ls1]
   + shortage[1,offpeak,HY,ls2] + shortage[1,offpeak,HY,ls3]
   + shortage[1,offpeak,HY,ls4] + shortage[1,offpeak,HY,ls5]
   + powerflow[1,offpeak,NI,HY] - powerflow[1,offpeak,HY,NI]
   - powerflow[1,offpeak,HY,SI] + powerflow[1,offpeak,SI,HY] = 142.92
 supplydemandbalance[1,offpeak,NI]: generation[1,offpeak,Huntly_U6]
   + generation[1,offpeak,Huntly_U5] + generation[1,offpeak,TCC]
   + generation[1,offpeak,NI_Thermal] + generation[1,offpeak,Arapuni]
   + generation[1,offpeak,Aratiatia] + generation[1,offpeak,Atiamuri]
   + generation[1,offpeak,Karapiro] + generation[1,offpeak,Maraetai]
   + generation[1,offpeak,Ohakuri] + generation[1,offpeak,Waipapa]
   + generation[1,offpeak,Whakamaru] + generation[1,offpeak,Mangahao]
   + generation[1,offpeak,Matahina] + generation[1,offpeak,Rangipo]
   + generation[1,offpeak,Tokaanu] + generation[1,offpeak,Waikaremoana]
   + shortage[1,offpeak,NI,ls1] + shortage[1,offpeak,NI,ls2]
   + shortage[1,offpeak,NI,ls3] + shortage[1,offpeak,NI,ls4]
   + shortage[1,offpeak,NI,ls5] - powerflow[1,offpeak,NI,HY]
   + powerflow[1,offpeak,HY,NI] = 751.939
 supplydemandbalance[1,offpeak,SI]: generation[1,offpeak,SI_Thermal]
   + generation[1,offpeak,Aviemore] + generation[1,offpeak,Benmore]
   + generation[1,offpeak,Ohau_A] + generation[1,offpeak,Ohau_B]
   + generation[1,offpeak,Ohau_C] + generation[1,offpeak,Tekapo_B]
   + generation[1,offpeak,Tekapo_A] + generation[1,offpeak,Waitaki]
   + generation[1,offpeak,Clyde_220kV] + generation[1,offpeak,Roxburgh]
   + generation[1,offpeak,Manapouri] + generation[1,offpeak,Cobb]
   + generation[1,offpeak,Coleridge] + shortage[1,offpeak,SI,ls1]
   + shortage[1,offpeak,SI,ls2] + shortage[1,offpeak,SI,ls3]
   + shortage[1,offpeak,SI,ls4] + shortage[1,offpeak,SI,ls5]
   + powerflow[1,offpeak,HY,SI] - powerflow[1,offpeak,SI,HY] = 1476.693
 hydrogenerationconversion[1,peak,Arapuni]: generation[1,peak,Arapuni]
   - 0.4619 turbineH20flow[1,peak,Arapuni] = 0
 hydrogenerationconversion[1,peak,Aratiatia]: generation[1,peak,Aratiatia]
   - 0.2841 turbineH20flow[1,peak,Aratiatia] = 0
 hydrogenerationconversion[1,peak,Atiamuri]: generation[1,peak,Atiamuri]
   - 0.1957 turbineH20flow[1,peak,Atiamuri] = 0
 hydrogenerationconversion[1,peak,Karapiro]: generation[1,peak,Karapiro]
   - 0.2639 turbineH20flow[1,peak,Karapiro] = 0
 hydrogenerationconversion[1,peak,Maraetai]: generation[1,peak,Maraetai]
   - 0.5263 turbineH20flow[1,peak,Maraetai] = 0
 hydrogenerationconversion[1,peak,Ohakuri]: generation[1,peak,Ohakuri]
   - 0.2841 turbineH20flow[1,peak,Ohakuri] = 0
 hydrogenerationconversion[1,peak,Waipapa]: generation[1,peak,Waipapa]
   - 0.1385 turbineH20flow[1,peak,Waipapa] = 0
 hydrogenerationconversion[1,peak,Whakamaru]: generation[1,peak,Whakamaru]
   - 0.3165 turbineH20flow[1,peak,Whakamaru] = 0
 hydrogenerationconversion[1,peak,Aviemore]: generation[1,peak,Aviemore]
   - 0.3101 turbineH20flow[1,peak,Aviemore] = 0
 hydrogenerationconversion[1,peak,Benmore]: generation[1,peak,Benmore]
   - 0.8177 turbineH20flow[1,peak,Benmore] = 0
 hydrogenerationconversion[1,peak,Ohau_A]: generation[1,peak,Ohau_A]
   - 0.5005 turbineH20flow[1,peak,Ohau_A] = 0
 hydrogenerationconversion[1,peak,Ohau_B]: generation[1,peak,Ohau_B]
   - 0.4167 turbineH20flow[1,peak,Ohau_B] = 0
 hydrogenerationconversion[1,peak,Ohau_C]: generation[1,peak,Ohau_C]
   - 0.4167 turbineH20flow[1,peak,Ohau_C] = 0
 hydrogenerationconversion[1,peak,Tekapo_B]: generation[1,peak,Tekapo_B]
   - 1.2847 turbineH20flow[1,peak,Tekapo_B] = 0
 hydrogenerationconversion[1,peak,Tekapo_A]: generation[1,peak,Tekapo_A]
   - 0.2322 turbineH20flow[1,peak,Tekapo_A] = 0
 hydrogenerationconversion[1,peak,Waitaki]: generation[1,peak,Waitaki]
   - 0.1622 turbineH20flow[1,peak,Waitaki] = 0
 hydrogenerationconversion[1,peak,Clyde_220kV]:
   generation[1,peak,Clyde_220kV]
   - 0.5181 turbineH20flow[1,peak,Clyde_220kV] = 0
 hydrogenerationconversion[1,peak,Roxburgh]: generation[1,peak,Roxburgh]
   - 0.4016 turbineH20flow[1,peak,Roxburgh] = 0
 hydrogenerationconversion[1,peak,Manapouri]: generation[1,peak,Manapouri]
   - 1.5314 turbineH20flow[1,peak,Manapouri] = 0
 hydrogenerationconversion[1,peak,Cobb]: generation[1,peak,Cobb]
   - 4.405 turbineH20flow[1,peak,Cobb] = 0
 hydrogenerationconversion[1,peak,Coleridge]: generation[1,peak,Coleridge]
   - 1.009 turbineH20flow[1,peak,Coleridge] = 0
 hydrogenerationconversion[1,peak,Mangahao]: generation[1,peak,Mangahao]
   - 2.53 turbineH20flow[1,peak,Mangahao] = 0
 hydrogenerationconversion[1,peak,Matahina]: generation[1,peak,Matahina]
   - 0.595 turbineH20flow[1,peak,Matahina] = 0
 hydrogenerationconversion[1,peak,Rangipo]: generation[1,peak,Rangipo]
   - 1.96 turbineH20flow[1,peak,Rangipo] = 0
 hydrogenerationconversion[1,peak,Tokaanu]: generation[1,peak,Tokaanu]
   - 1.75 turbineH20flow[1,peak,Tokaanu] = 0
 hydrogenerationconversion[1,peak,Waikaremoana]:
   generation[1,peak,Waikaremoana]
   - 3.535 turbineH20flow[1,peak,Waikaremoana] = 0
 hydrogenerationconversion[1,shoulder,Arapuni]:
   generation[1,shoulder,Arapuni]
   - 0.4619 turbineH20flow[1,shoulder,Arapuni] = 0
 hydrogenerationconversion[1,shoulder,Aratiatia]:
   generation[1,shoulder,Aratiatia]
   - 0.2841 turbineH20flow[1,shoulder,Aratiatia] = 0
 hydrogenerationconversion[1,shoulder,Atiamuri]:
   generation[1,shoulder,Atiamuri]
   - 0.1957 turbineH20flow[1,shoulder,Atiamuri] = 0
 hydrogenerationconversion[1,shoulder,Karapiro]:
   generation[1,shoulder,Karapiro]
   - 0.2639 turbineH20flow[1,shoulder,Karapiro] = 0
 hydrogenerationconversion[1,shoulder,Maraetai]:
   generation[1,shoulder,Maraetai]
   - 0.5263 turbineH20flow[1,shoulder,Maraetai] = 0
 hydrogenerationconversion[1,shoulder,Ohakuri]:
   generation[1,shoulder,Ohakuri]
   - 0.2841 turbineH20flow[1,shoulder,Ohakuri] = 0
 hydrogenerationconversion[1,shoulder,Waipapa]:
   generation[1,shoulder,Waipapa]
   - 0.1385 turbineH20flow[1,shoulder,Waipapa] = 0
 hydrogenerationconversion[1,shoulder,Whakamaru]:
   generation[1,shoulder,Whakamaru]
   - 0.3165 turbineH20flow[1,shoulder,Whakamaru] = 0
 hydrogenerationconversion[1,shoulder,Aviemore]:
   generation[1,shoulder,Aviemore]
   - 0.3101 turbineH20flow[1,shoulder,Aviemore] = 0
 hydrogenerationconversion[1,shoulder,Benmore]:
   generation[1,shoulder,Benmore]
   - 0.8177 turbineH20flow[1,shoulder,Benmore] = 0
 hydrogenerationconversion[1,shoulder,Ohau_A]:
   generation[1,shoulder,Ohau_A] - 0.5005 turbineH20flow[1,shoulder,Ohau_A]
   = 0
 hydrogenerationconversion[1,shoulder,Ohau_B]:
   generation[1,shoulder,Ohau_B] - 0.4167 turbineH20flow[1,shoulder,Ohau_B]
   = 0
 hydrogenerationconversion[1,shoulder,Ohau_C]:
   generation[1,shoulder,Ohau_C] - 0.4167 turbineH20flow[1,shoulder,Ohau_C]
   = 0
 hydrogenerationconversion[1,shoulder,Tekapo_B]:
   generation[1,shoulder,Tekapo_B]
   - 1.2847 turbineH20flow[1,shoulder,Tekapo_B] = 0
 hydrogenerationconversion[1,shoulder,Tekapo_A]:
   generation[1,shoulder,Tekapo_A]
   - 0.2322 turbineH20flow[1,shoulder,Tekapo_A] = 0
 hydrogenerationconversion[1,shoulder,Waitaki]:
   generation[1,shoulder,Waitaki]
   - 0.1622 turbineH20flow[1,shoulder,Waitaki] = 0
 hydrogenerationconversion[1,shoulder,Clyde_220kV]:
   generation[1,shoulder,Clyde_220kV]
   - 0.5181 turbineH20flow[1,shoulder,Clyde_220kV] = 0
 hydrogenerationconversion[1,shoulder,Roxburgh]:
   generation[1,shoulder,Roxburgh]
   - 0.4016 turbineH20flow[1,shoulder,Roxburgh] = 0
 hydrogenerationconversion[1,shoulder,Manapouri]:
   generation[1,shoulder,Manapouri]
   - 1.5314 turbineH20flow[1,shoulder,Manapouri] = 0
 hydrogenerationconversion[1,shoulder,Cobb]: generation[1,shoulder,Cobb]
   - 4.405 turbineH20flow[1,shoulder,Cobb] = 0
 hydrogenerationconversion[1,shoulder,Coleridge]:
   generation[1,shoulder,Coleridge]
   - 1.009 turbineH20flow[1,shoulder,Coleridge] = 0
 hydrogenerationconversion[1,shoulder,Mangahao]:
   generation[1,shoulder,Mangahao]
   - 2.53 turbineH20flow[1,shoulder,Mangahao] = 0
 hydrogenerationconversion[1,shoulder,Matahina]:
   generation[1,shoulder,Matahina]
   - 0.595 turbineH20flow[1,shoulder,Matahina] = 0
 hydrogenerationconversion[1,shoulder,Rangipo]:
   generation[1,shoulder,Rangipo] - 1.96 turbineH20flow[1,shoulder,Rangipo]
   = 0
 hydrogenerationconversion[1,shoulder,Tokaanu]:
   generation[1,shoulder,Tokaanu] - 1.75 turbineH20flow[1,shoulder,Tokaanu]
   = 0
 hydrogenerationconversion[1,shoulder,Waikaremoana]:
   generation[1,shoulder,Waikaremoana]
   - 3.535 turbineH20flow[1,shoulder,Waikaremoana] = 0
 hydrogenerationconversion[1,offpeak,Arapuni]:
   generation[1,offpeak,Arapuni] - 0.4619 turbineH20flow[1,offpeak,Arapuni]
   = 0
 hydrogenerationconversion[1,offpeak,Aratiatia]:
   generation[1,offpeak,Aratiatia]
   - 0.2841 turbineH20flow[1,offpeak,Aratiatia] = 0
 hydrogenerationconversion[1,offpeak,Atiamuri]:
   generation[1,offpeak,Atiamuri]
   - 0.1957 turbineH20flow[1,offpeak,Atiamuri] = 0
 hydrogenerationconversion[1,offpeak,Karapiro]:
   generation[1,offpeak,Karapiro]
   - 0.2639 turbineH20flow[1,offpeak,Karapiro] = 0
 hydrogenerationconversion[1,offpeak,Maraetai]:
   generation[1,offpeak,Maraetai]
   - 0.5263 turbineH20flow[1,offpeak,Maraetai] = 0
 hydrogenerationconversion[1,offpeak,Ohakuri]:
   generation[1,offpeak,Ohakuri] - 0.2841 turbineH20flow[1,offpeak,Ohakuri]
   = 0
 hydrogenerationconversion[1,offpeak,Waipapa]:
   generation[1,offpeak,Waipapa] - 0.1385 turbineH20flow[1,offpeak,Waipapa]
   = 0
 hydrogenerationconversion[1,offpeak,Whakamaru]:
   generation[1,offpeak,Whakamaru]
   - 0.3165 turbineH20flow[1,offpeak,Whakamaru] = 0
 hydrogenerationconversion[1,offpeak,Aviemore]:
   generation[1,offpeak,Aviemore]
   - 0.3101 turbineH20flow[1,offpeak,Aviemore] = 0
 hydrogenerationconversion[1,offpeak,Benmore]:
   generation[1,offpeak,Benmore] - 0.8177 turbineH20flow[1,offpeak,Benmore]
   = 0
 hydrogenerationconversion[1,offpeak,Ohau_A]: generation[1,offpeak,Ohau_A]
   - 0.5005 turbineH20flow[1,offpeak,Ohau_A] = 0
 hydrogenerationconversion[1,offpeak,Ohau_B]: generation[1,offpeak,Ohau_B]
   - 0.4167 turbineH20flow[1,offpeak,Ohau_B] = 0
 hydrogenerationconversion[1,offpeak,Ohau_C]: generation[1,offpeak,Ohau_C]
   - 0.4167 turbineH20flow[1,offpeak,Ohau_C] = 0
 hydrogenerationconversion[1,offpeak,Tekapo_B]:
   generation[1,offpeak,Tekapo_B]
   - 1.2847 turbineH20flow[1,offpeak,Tekapo_B] = 0
 hydrogenerationconversion[1,offpeak,Tekapo_A]:
   generation[1,offpeak,Tekapo_A]
   - 0.2322 turbineH20flow[1,offpeak,Tekapo_A] = 0
 hydrogenerationconversion[1,offpeak,Waitaki]:
   generation[1,offpeak,Waitaki] - 0.1622 turbineH20flow[1,offpeak,Waitaki]
   = 0
 hydrogenerationconversion[1,offpeak,Clyde_220kV]:
   generation[1,offpeak,Clyde_220kV]
   - 0.5181 turbineH20flow[1,offpeak,Clyde_220kV] = 0
 hydrogenerationconversion[1,offpeak,Roxburgh]:
   generation[1,offpeak,Roxburgh]
   - 0.4016 turbineH20flow[1,offpeak,Roxburgh] = 0
 hydrogenerationconversion[1,offpeak,Manapouri]:
   generation[1,offpeak,Manapouri]
   - 1.5314 turbineH20flow[1,offpeak,Manapouri] = 0
 hydrogenerationconversion[1,offpeak,Cobb]: generation[1,offpeak,Cobb]
   - 4.405 turbineH20flow[1,offpeak,Cobb] = 0
 hydrogenerationconversion[1,offpeak,Coleridge]:
   generation[1,offpeak,Coleridge]
   - 1.009 turbineH20flow[1,offpeak,Coleridge] = 0
 hydrogenerationconversion[1,offpeak,Mangahao]:
   generation[1,offpeak,Mangahao] - 2.53 turbineH20flow[1,offpeak,Mangahao]
   = 0
 hydrogenerationconversion[1,offpeak,Matahina]:
   generation[1,offpeak,Matahina]
   - 0.595 turbineH20flow[1,offpeak,Matahina] = 0
 hydrogenerationconversion[1,offpeak,Rangipo]:
   generation[1,offpeak,Rangipo] - 1.96 turbineH20flow[1,offpeak,Rangipo]
   = 0
 hydrogenerationconversion[1,offpeak,Tokaanu]:
   generation[1,offpeak,Tokaanu] - 1.75 turbineH20flow[1,offpeak,Tokaanu]
   = 0
 hydrogenerationconversion[1,offpeak,Waikaremoana]:
   generation[1,offpeak,Waikaremoana]
   - 3.535 turbineH20flow[1,offpeak,Waikaremoana] = 0
 maxhydroflow[1,peak,Lake_Hawea,Lake_Dunstan]:
   arcflow[1,peak,Lake_Hawea,Lake_Dunstan]
   - arcflowdeficit[1,peak,Lake_Hawea,Lake_Dunstan] <= 200
 maxhydroflow[1,peak,Lake_Tekapo,Lake_Scott]:
   arcflow[1,peak,Lake_Tekapo,Lake_Scott]
   - arcflowdeficit[1,peak,Lake_Tekapo,Lake_Scott] <= 850
 maxhydroflow[1,peak,Lake_Scott,Tekapo_canal_head]:
   arcflow[1,peak,Lake_Scott,Tekapo_canal_head]
   - arcflowdeficit[1,peak,Lake_Scott,Tekapo_canal_head] <= 130
 maxhydroflow[1,peak,Lake_Scott,Lake_Benmore]:
   arcflow[1,peak,Lake_Scott,Lake_Benmore]
   - arcflowdeficit[1,peak,Lake_Scott,Lake_Benmore] <= 600
 maxhydroflow[1,peak,Tekapo_canal_head,Tekapo_canal_tail]:
   arcflow[1,peak,Tekapo_canal_head,Tekapo_canal_tail]
   - arcflowdeficit[1,peak,Tekapo_canal_head,Tekapo_canal_tail] <= 130
 maxhydroflow[1,peak,Lake_Pukaki,Lake_Benmore]:
   arcflow[1,peak,Lake_Pukaki,Lake_Benmore]
   - arcflowdeficit[1,peak,Lake_Pukaki,Lake_Benmore] <= 3400
 maxhydroflow[1,peak,Lake_Pukaki,Pukaki_Ohau_canal_junction]:
   arcflow[1,peak,Lake_Pukaki,Pukaki_Ohau_canal_junction]
   - arcflowdeficit[1,peak,Lake_Pukaki,Pukaki_Ohau_canal_junction] <= 560
 maxhydroflow[1,peak,Lake_Ohau,Pukaki_Ohau_canal_junction]:
   arcflow[1,peak,Lake_Ohau,Pukaki_Ohau_canal_junction]
   - arcflowdeficit[1,peak,Lake_Ohau,Pukaki_Ohau_canal_junction] <= 200
 maxhydroflow[1,peak,Lake_Ruataniwha,Lake_Benmore]:
   arcflow[1,peak,Lake_Ruataniwha,Lake_Benmore]
   - arcflowdeficit[1,peak,Lake_Ruataniwha,Lake_Benmore] <= 1740
 maxhydroflow[1,shoulder,Lake_Hawea,Lake_Dunstan]:
   arcflow[1,shoulder,Lake_Hawea,Lake_Dunstan]
   - arcflowdeficit[1,shoulder,Lake_Hawea,Lake_Dunstan] <= 200
 maxhydroflow[1,shoulder,Lake_Tekapo,Lake_Scott]:
   arcflow[1,shoulder,Lake_Tekapo,Lake_Scott]
   - arcflowdeficit[1,shoulder,Lake_Tekapo,Lake_Scott] <= 850
 maxhydroflow[1,shoulder,Lake_Scott,Tekapo_canal_head]:
   arcflow[1,shoulder,Lake_Scott,Tekapo_canal_head]
   - arcflowdeficit[1,shoulder,Lake_Scott,Tekapo_canal_head] <= 130
 maxhydroflow[1,shoulder,Lake_Scott,Lake_Benmore]:
   arcflow[1,shoulder,Lake_Scott,Lake_Benmore]
   - arcflowdeficit[1,shoulder,Lake_Scott,Lake_Benmore] <= 600
 maxhydroflow[1,shoulder,Tekapo_canal_head,Tekapo_canal_tail]:
   arcflow[1,shoulder,Tekapo_canal_head,Tekapo_canal_tail]
   - arcflowdeficit[1,shoulder,Tekapo_canal_head,Tekapo_canal_tail] <= 130
 maxhydroflow[1,shoulder,Lake_Pukaki,Lake_Benmore]:
   arcflow[1,shoulder,Lake_Pukaki,Lake_Benmore]
   - arcflowdeficit[1,shoulder,Lake_Pukaki,Lake_Benmore] <= 3400
 maxhydroflow[1,shoulder,Lake_Pukaki,Pukaki_Ohau_canal_junction]:
   arcflow[1,shoulder,Lake_Pukaki,Pukaki_Ohau_canal_junction]
   - arcflowdeficit[1,shoulder,Lake_Pukaki,Pukaki_Ohau_canal_junction]
   <= 560
 maxhydroflow[1,shoulder,Lake_Ohau,Pukaki_Ohau_canal_junction]:
   arcflow[1,shoulder,Lake_Ohau,Pukaki_Ohau_canal_junction]
   - arcflowdeficit[1,shoulder,Lake_Ohau,Pukaki_Ohau_canal_junction]
   <= 200
 maxhydroflow[1,shoulder,Lake_Ruataniwha,Lake_Benmore]:
   arcflow[1,shoulder,Lake_Ruataniwha,Lake_Benmore]
   - arcflowdeficit[1,shoulder,Lake_Ruataniwha,Lake_Benmore] <= 1740
 maxhydroflow[1,offpeak,Lake_Hawea,Lake_Dunstan]:
   arcflow[1,offpeak,Lake_Hawea,Lake_Dunstan]
   - arcflowdeficit[1,offpeak,Lake_Hawea,Lake_Dunstan] <= 200
 maxhydroflow[1,offpeak,Lake_Tekapo,Lake_Scott]:
   arcflow[1,offpeak,Lake_Tekapo,Lake_Scott]
   - arcflowdeficit[1,offpeak,Lake_Tekapo,Lake_Scott] <= 850
 maxhydroflow[1,offpeak,Lake_Scott,Tekapo_canal_head]:
   arcflow[1,offpeak,Lake_Scott,Tekapo_canal_head]
   - arcflowdeficit[1,offpeak,Lake_Scott,Tekapo_canal_head] <= 130
 maxhydroflow[1,offpeak,Lake_Scott,Lake_Benmore]:
   arcflow[1,offpeak,Lake_Scott,Lake_Benmore]
   - arcflowdeficit[1,offpeak,Lake_Scott,Lake_Benmore] <= 600
 maxhydroflow[1,offpeak,Tekapo_canal_head,Tekapo_canal_tail]:
   arcflow[1,offpeak,Tekapo_canal_head,Tekapo_canal_tail]
   - arcflowdeficit[1,offpeak,Tekapo_canal_head,Tekapo_canal_tail] <= 130
 maxhydroflow[1,offpeak,Lake_Pukaki,Lake_Benmore]:
   arcflow[1,offpeak,Lake_Pukaki,Lake_Benmore]
   - arcflowdeficit[1,offpeak,Lake_Pukaki,Lake_Benmore] <= 3400
 maxhydroflow[1,offpeak,Lake_Pukaki,Pukaki_Ohau_canal_junction]:
   arcflow[1,offpeak,Lake_Pukaki,Pukaki_Ohau_canal_junction]
   - arcflowdeficit[1,offpeak,Lake_Pukaki,Pukaki_Ohau_canal_junction]
   <= 560
 maxhydroflow[1,offpeak,Lake_Ohau,Pukaki_Ohau_canal_junction]:
   arcflow[1,offpeak,Lake_Ohau,Pukaki_Ohau_canal_junction]
   - arcflowdeficit[1,offpeak,Lake_Ohau,Pukaki_Ohau_canal_junction] <= 200
 maxhydroflow[1,offpeak,Lake_Ruataniwha,Lake_Benmore]:
   arcflow[1,offpeak,Lake_Ruataniwha,Lake_Benmore]
   - arcflowdeficit[1,offpeak,Lake_Ruataniwha,Lake_Benmore] <= 1740
 minhydroflow[1,peak,Lake_Hawea,Lake_Dunstan]:
   arcflow[1,peak,Lake_Hawea,Lake_Dunstan]
   + arcflowsurplus[1,peak,Lake_Hawea,Lake_Dunstan] >= 10
 minhydroflow[1,peak,Clyde_tail,Lake_Roxburgh]:
   arcflow[1,peak,Clyde_tail,Lake_Roxburgh]
   + arcflowsurplus[1,peak,Clyde_tail,Lake_Roxburgh] >= 66.25
 minhydroflow[1,peak,Roxburgh_tail,SEA]: arcflow[1,peak,Roxburgh_tail,SEA]
   + arcflowsurplus[1,peak,Roxburgh_tail,SEA] >= 250
 minhydroflow[1,peak,Lake_Taupo,Lake_Aratiatia]:
   arcflow[1,peak,Lake_Taupo,Lake_Aratiatia]
   + arcflowsurplus[1,peak,Lake_Taupo,Lake_Aratiatia] >= 50
 minhydroflow[1,peak,Karapiro_tail,SEA]: arcflow[1,peak,Karapiro_tail,SEA]
   + arcflowsurplus[1,peak,Karapiro_tail,SEA] >= 148
 minhydroflow[1,peak,Lake_Ohau,Lake_Ruataniwha]:
   arcflow[1,peak,Lake_Ohau,Lake_Ruataniwha]
   + arcflowsurplus[1,peak,Lake_Ohau,Lake_Ruataniwha] >= 10
 minhydroflow[1,peak,Waitaki_tail,SEA]: arcflow[1,peak,Waitaki_tail,SEA]
   + arcflowsurplus[1,peak,Waitaki_tail,SEA] >= 170.5
 minhydroflow[1,shoulder,Lake_Hawea,Lake_Dunstan]:
   arcflow[1,shoulder,Lake_Hawea,Lake_Dunstan]
   + arcflowsurplus[1,shoulder,Lake_Hawea,Lake_Dunstan] >= 10
 minhydroflow[1,shoulder,Clyde_tail,Lake_Roxburgh]:
   arcflow[1,shoulder,Clyde_tail,Lake_Roxburgh]
   + arcflowsurplus[1,shoulder,Clyde_tail,Lake_Roxburgh] >= 66.25
 minhydroflow[1,shoulder,Roxburgh_tail,SEA]:
   arcflow[1,shoulder,Roxburgh_tail,SEA]
   + arcflowsurplus[1,shoulder,Roxburgh_tail,SEA] >= 250
 minhydroflow[1,shoulder,Lake_Taupo,Lake_Aratiatia]:
   arcflow[1,shoulder,Lake_Taupo,Lake_Aratiatia]
   + arcflowsurplus[1,shoulder,Lake_Taupo,Lake_Aratiatia] >= 50
 minhydroflow[1,shoulder,Karapiro_tail,SEA]:
   arcflow[1,shoulder,Karapiro_tail,SEA]
   + arcflowsurplus[1,shoulder,Karapiro_tail,SEA] >= 148
 minhydroflow[1,shoulder,Lake_Ohau,Lake_Ruataniwha]:
   arcflow[1,shoulder,Lake_Ohau,Lake_Ruataniwha]
   + arcflowsurplus[1,shoulder,Lake_Ohau,Lake_Ruataniwha] >= 10
 minhydroflow[1,shoulder,Waitaki_tail,SEA]:
   arcflow[1,shoulder,Waitaki_tail,SEA]
   + arcflowsurplus[1,shoulder,Waitaki_tail,SEA] >= 170.5
 minhydroflow[1,offpeak,Lake_Hawea,Lake_Dunstan]:
   arcflow[1,offpeak,Lake_Hawea,Lake_Dunstan]
   + arcflowsurplus[1,offpeak,Lake_Hawea,Lake_Dunstan] >= 10
 minhydroflow[1,offpeak,Clyde_tail,Lake_Roxburgh]:
   arcflow[1,offpeak,Clyde_tail,Lake_Roxburgh]
   + arcflowsurplus[1,offpeak,Clyde_tail,Lake_Roxburgh] >= 66.25
 minhydroflow[1,offpeak,Roxburgh_tail,SEA]:
   arcflow[1,offpeak,Roxburgh_tail,SEA]
   + arcflowsurplus[1,offpeak,Roxburgh_tail,SEA] >= 250
 minhydroflow[1,offpeak,Lake_Taupo,Lake_Aratiatia]:
   arcflow[1,offpeak,Lake_Taupo,Lake_Aratiatia]
   + arcflowsurplus[1,offpeak,Lake_Taupo,Lake_Aratiatia] >= 50
 minhydroflow[1,offpeak,Karapiro_tail,SEA]:
   arcflow[1,offpeak,Karapiro_tail,SEA]
   + arcflowsurplus[1,offpeak,Karapiro_tail,SEA] >= 148
 minhydroflow[1,offpeak,Lake_Ohau,Lake_Ruataniwha]:
   arcflow[1,offpeak,Lake_Ohau,Lake_Ruataniwha]
   + arcflowsurplus[1,offpeak,Lake_Ohau,Lake_Ruataniwha] >= 10
 minhydroflow[1,offpeak,Waitaki_tail,SEA]:
   arcflow[1,offpeak,Waitaki_tail,SEA]
   + arcflowsurplus[1,offpeak,Waitaki_tail,SEA] >= 170.5
 hydrojunctionconservation[1,Lake_Wanaka]:
   72000 arcflow[1,peak,Lake_Wanaka,Lake_Dunstan]
   + 216000 arcflow[1,shoulder,Lake_Wanaka,Lake_Dunstan]
   + 316800 arcflow[1,offpeak,Lake_Wanaka,Lake_Dunstan] = 0
 hydrojunctionconservation[1,Clyde_tail]:
   - 72000 turbineH20flow[1,peak,Clyde_220kV]
   - 216000 turbineH20flow[1,shoulder,Clyde_220kV]
   - 316800 turbineH20flow[1,offpeak,Clyde_220kV]
   - 72000 spilledH20flow[1,peak,Clyde_220kV]
   - 216000 spilledH20flow[1,shoulder,Clyde_220kV]
   - 316800 spilledH20flow[1,offpeak,Clyde_220kV]
   + 72000 arcflow[1,peak,Clyde_tail,Lake_Roxburgh]
   + 216000 arcflow[1,shoulder,Clyde_tail,Lake_Roxburgh]
   + 316800 arcflow[1,offpeak,Clyde_tail,Lake_Roxburgh] = 0
 hydrojunctionconservation[1,Roxburgh_tail]:
   - 72000 turbineH20flow[1,peak,Roxburgh]
   - 216000 turbineH20flow[1,shoulder,Roxburgh]
   - 316800 turbineH20flow[1,offpeak,Roxburgh]
   - 72000 spilledH20flow[1,peak,Roxburgh]
   - 216000 spilledH20flow[1,shoulder,Roxburgh]
   - 316800 spilledH20flow[1,offpeak,Roxburgh]
   + 72000 arcflow[1,peak,Roxburgh_tail,SEA]
   + 216000 arcflow[1,shoulder,Roxburgh_tail,SEA]
   + 316800 arcflow[1,offpeak,Roxburgh_tail,SEA] = 0
 hydrojunctionconservation[1,Karapiro_tail]:
   - 72000 turbineH20flow[1,peak,Karapiro]
   - 216000 turbineH20flow[1,shoulder,Karapiro]
   - 316800 turbineH20flow[1,offpeak,Karapiro]
   - 72000 spilledH20flow[1,peak,Karapiro]
   - 216000 spilledH20flow[1,shoulder,Karapiro]
   - 316800 spilledH20flow[1,offpeak,Karapiro]
   + 72000 arcflow[1,peak,Karapiro_tail,SEA]
   + 216000 arcflow[1,shoulder,Karapiro_tail,SEA]
   + 316800 arcflow[1,offpeak,Karapiro_tail,SEA] = 0
 hydrojunctionconservation[1,Lake_Scott]:
   - 72000 arcflow[1,peak,Lake_Tekapo,Lake_Scott]
   + 72000 arcflow[1,peak,Lake_Scott,Tekapo_canal_head]
   + 72000 arcflow[1,peak,Lake_Scott,Lake_Benmore]
   - 216000 arcflow[1,shoulder,Lake_Tekapo,Lake_Scott]
   + 216000 arcflow[1,shoulder,Lake_Scott,Tekapo_canal_head]
   + 216000 arcflow[1,shoulder,Lake_Scott,Lake_Benmore]
   - 316800 arcflow[1,offpeak,Lake_Tekapo,Lake_Scott]
   + 316800 arcflow[1,offpeak,Lake_Scott,Tekapo_canal_head]
   + 316800 arcflow[1,offpeak,Lake_Scott,Lake_Benmore] = 0
 hydrojunctionconservation[1,Tekapo_canal_head]:
   - 72000 turbineH20flow[1,peak,Tekapo_A]
   - 216000 turbineH20flow[1,shoulder,Tekapo_A]
   - 316800 turbineH20flow[1,offpeak,Tekapo_A]
   - 72000 spilledH20flow[1,peak,Tekapo_A]
   - 216000 spilledH20flow[1,shoulder,Tekapo_A]
   - 316800 spilledH20flow[1,offpeak,Tekapo_A]
   - 72000 arcflow[1,peak,Lake_Scott,Tekapo_canal_head]
   + 72000 arcflow[1,peak,Tekapo_canal_head,Tekapo_canal_tail]
   - 216000 arcflow[1,shoulder,Lake_Scott,Tekapo_canal_head]
   + 216000 arcflow[1,shoulder,Tekapo_canal_head,Tekapo_canal_tail]
   - 316800 arcflow[1,offpeak,Lake_Scott,Tekapo_canal_head]
   + 316800 arcflow[1,offpeak,Tekapo_canal_head,Tekapo_canal_tail] = 0
 hydrojunctionconservation[1,Lake_Ruataniwha]:
   - 72000 turbineH20flow[1,peak,Ohau_A]
   + 72000 turbineH20flow[1,peak,Ohau_B]
   - 216000 turbineH20flow[1,shoulder,Ohau_A]
   + 216000 turbineH20flow[1,shoulder,Ohau_B]
   - 316800 turbineH20flow[1,offpeak,Ohau_A]
   + 316800 turbineH20flow[1,offpeak,Ohau_B]
   - 72000 spilledH20flow[1,peak,Ohau_A]
   + 72000 spilledH20flow[1,peak,Ohau_B]
   - 216000 spilledH20flow[1,shoulder,Ohau_A]
   + 216000 spilledH20flow[1,shoulder,Ohau_B]
   - 316800 spilledH20flow[1,offpeak,Ohau_A]
   + 316800 spilledH20flow[1,offpeak,Ohau_B]
   - 72000 arcflow[1,peak,Lake_Ohau,Lake_Ruataniwha]
   + 72000 arcflow[1,peak,Lake_Ruataniwha,Lake_Benmore]
   - 216000 arcflow[1,shoulder,Lake_Ohau,Lake_Ruataniwha]
   + 216000 arcflow[1,shoulder,Lake_Ruataniwha,Lake_Benmore]
   - 316800 arcflow[1,offpeak,Lake_Ohau,Lake_Ruataniwha]
   + 316800 arcflow[1,offpeak,Lake_Ruataniwha,Lake_Benmore] = 0
 hydrojunctionconservation[1,Ohau_BC_canal]:
   - 72000 turbineH20flow[1,peak,Ohau_B]
   + 72000 turbineH20flow[1,peak,Ohau_C]
   - 216000 turbineH20flow[1,shoulder,Ohau_B]
   + 216000 turbineH20flow[1,shoulder,Ohau_C]
   - 316800 turbineH20flow[1,offpeak,Ohau_B]
   + 316800 turbineH20flow[1,offpeak,Ohau_C]
   - 72000 spilledH20flow[1,peak,Ohau_B]
   + 72000 spilledH20flow[1,peak,Ohau_C]
   - 216000 spilledH20flow[1,shoulder,Ohau_B]
   + 216000 spilledH20flow[1,shoulder,Ohau_C]
   - 316800 spilledH20flow[1,offpeak,Ohau_B]
   + 316800 spilledH20flow[1,offpeak,Ohau_C]
   + 72000 arcflow[1,peak,Ohau_BC_canal,Lake_Benmore]
   + 216000 arcflow[1,shoulder,Ohau_BC_canal,Lake_Benmore]
   + 316800 arcflow[1,offpeak,Ohau_BC_canal,Lake_Benmore] = 0
 hydrojunctionconservation[1,Waitaki_tail]:
   - 72000 turbineH20flow[1,peak,Waitaki]
   - 216000 turbineH20flow[1,shoulder,Waitaki]
   - 316800 turbineH20flow[1,offpeak,Waitaki]
   - 72000 spilledH20flow[1,peak,Waitaki]
   - 216000 spilledH20flow[1,shoulder,Waitaki]
   - 316800 spilledH20flow[1,offpeak,Waitaki]
   + 72000 arcflow[1,peak,Waitaki_tail,SEA]
   + 216000 arcflow[1,shoulder,Waitaki_tail,SEA]
   + 316800 arcflow[1,offpeak,Waitaki_tail,SEA] = 0
 hydrojunctionconservation[1,Lake_Dunstan]:
   72000 turbineH20flow[1,peak,Clyde_220kV]
   + 216000 turbineH20flow[1,shoulder,Clyde_220kV]
   + 316800 turbineH20flow[1,offpeak,Clyde_220kV]
   + 72000 spilledH20flow[1,peak,Clyde_220kV]
   + 216000 spilledH20flow[1,shoulder,Clyde_220kV]
   + 316800 spilledH20flow[1,offpeak,Clyde_220kV]
   - 72000 arcflow[1,peak,Lake_Wanaka,Lake_Dunstan]
   - 72000 arcflow[1,peak,Lake_Hawea,Lake_Dunstan]
   - 216000 arcflow[1,shoulder,Lake_Wanaka,Lake_Dunstan]
   - 216000 arcflow[1,shoulder,Lake_Hawea,Lake_Dunstan]
   - 316800 arcflow[1,offpeak,Lake_Wanaka,Lake_Dunstan]
   - 316800 arcflow[1,offpeak,Lake_Hawea,Lake_Dunstan] = 0
 hydrojunctionconservation[1,Lake_Roxburgh]:
   72000 turbineH20flow[1,peak,Roxburgh]
   + 216000 turbineH20flow[1,shoulder,Roxburgh]
   + 316800 turbineH20flow[1,offpeak,Roxburgh]
   + 72000 spilledH20flow[1,peak,Roxburgh]
   + 216000 spilledH20flow[1,shoulder,Roxburgh]
   + 316800 spilledH20flow[1,offpeak,Roxburgh]
   - 72000 arcflow[1,peak,Clyde_tail,Lake_Roxburgh]
   - 216000 arcflow[1,shoulder,Clyde_tail,Lake_Roxburgh]
   - 316800 arcflow[1,offpeak,Clyde_tail,Lake_Roxburgh] = 0
 hydrojunctionconservation[1,Lake_Aratiatia]:
   72000 turbineH20flow[1,peak,Aratiatia]
   + 216000 turbineH20flow[1,shoulder,Aratiatia]
   + 316800 turbineH20flow[1,offpeak,Aratiatia]
   + 72000 spilledH20flow[1,peak,Aratiatia]
   + 216000 spilledH20flow[1,shoulder,Aratiatia]
   + 316800 spilledH20flow[1,offpeak,Aratiatia]
   - 72000 arcflow[1,peak,Lake_Taupo,Lake_Aratiatia]
   - 216000 arcflow[1,shoulder,Lake_Taupo,Lake_Aratiatia]
   - 316800 arcflow[1,offpeak,Lake_Taupo,Lake_Aratiatia] = 0
 hydrojunctionconservation[1,Tekapo_canal_tail]:
   72000 turbineH20flow[1,peak,Tekapo_B]
   + 216000 turbineH20flow[1,shoulder,Tekapo_B]
   + 316800 turbineH20flow[1,offpeak,Tekapo_B]
   + 72000 spilledH20flow[1,peak,Tekapo_B]
   + 216000 spilledH20flow[1,shoulder,Tekapo_B]
   + 316800 spilledH20flow[1,offpeak,Tekapo_B]
   - 72000 arcflow[1,peak,Tekapo_canal_head,Tekapo_canal_tail]
   - 216000 arcflow[1,shoulder,Tekapo_canal_head,Tekapo_canal_tail]
   - 316800 arcflow[1,offpeak,Tekapo_canal_head,Tekapo_canal_tail] = 0
 hydrojunctionconservation[1,Pukaki_Ohau_canal_junction]:
   72000 turbineH20flow[1,peak,Ohau_A]
   + 216000 turbineH20flow[1,shoulder,Ohau_A]
   + 316800 turbineH20flow[1,offpeak,Ohau_A]
   + 72000 spilledH20flow[1,peak,Ohau_A]
   + 216000 spilledH20flow[1,shoulder,Ohau_A]
   + 316800 spilledH20flow[1,offpeak,Ohau_A]
   - 72000 arcflow[1,peak,Lake_Pukaki,Pukaki_Ohau_canal_junction]
   - 72000 arcflow[1,peak,Lake_Ohau,Pukaki_Ohau_canal_junction]
   - 216000 arcflow[1,shoulder,Lake_Pukaki,Pukaki_Ohau_canal_junction]
   - 216000 arcflow[1,shoulder,Lake_Ohau,Pukaki_Ohau_canal_junction]
   - 316800 arcflow[1,offpeak,Lake_Pukaki,Pukaki_Ohau_canal_junction]
   - 316800 arcflow[1,offpeak,Lake_Ohau,Pukaki_Ohau_canal_junction] = 0
 hydrojunctionconservation[1,Lake_Arapuni]:
   72000 turbineH20flow[1,peak,Arapuni]
   - 72000 turbineH20flow[1,peak,Waipapa]
   + 216000 turbineH20flow[1,shoulder,Arapuni]
   - 216000 turbineH20flow[1,shoulder,Waipapa]
   + 316800 turbineH20flow[1,offpeak,Arapuni]
   - 316800 turbineH20flow[1,offpeak,Waipapa]
   + 72000 spilledH20flow[1,peak,Arapuni]
   - 72000 spilledH20flow[1,peak,Waipapa]
   + 216000 spilledH20flow[1,shoulder,Arapuni]
   - 216000 spilledH20flow[1,shoulder,Waipapa]
   + 316800 spilledH20flow[1,offpeak,Arapuni]
   - 316800 spilledH20flow[1,offpeak,Waipapa] = 0
 hydrojunctionconservation[1,Lake_Atiamuri]:
   72000 turbineH20flow[1,peak,Atiamuri]
   - 72000 turbineH20flow[1,peak,Ohakuri]
   + 216000 turbineH20flow[1,shoulder,Atiamuri]
   - 216000 turbineH20flow[1,shoulder,Ohakuri]
   + 316800 turbineH20flow[1,offpeak,Atiamuri]
   - 316800 turbineH20flow[1,offpeak,Ohakuri]
   + 72000 spilledH20flow[1,peak,Atiamuri]
   - 72000 spilledH20flow[1,peak,Ohakuri]
   + 216000 spilledH20flow[1,shoulder,Atiamuri]
   - 216000 spilledH20flow[1,shoulder,Ohakuri]
   + 316800 spilledH20flow[1,offpeak,Atiamuri]
   - 316800 spilledH20flow[1,offpeak,Ohakuri] = 0
 hydrojunctionconservation[1,Lake_Karapiro]:
   - 72000 turbineH20flow[1,peak,Arapuni]
   + 72000 turbineH20flow[1,peak,Karapiro]
   - 216000 turbineH20flow[1,shoulder,Arapuni]
   + 216000 turbineH20flow[1,shoulder,Karapiro]
   - 316800 turbineH20flow[1,offpeak,Arapuni]
   + 316800 turbineH20flow[1,offpeak,Karapiro]
   - 72000 spilledH20flow[1,peak,Arapuni]
   + 72000 spilledH20flow[1,peak,Karapiro]
   - 216000 spilledH20flow[1,shoulder,Arapuni]
   + 216000 spilledH20flow[1,shoulder,Karapiro]
   - 316800 spilledH20flow[1,offpeak,Arapuni]
   + 316800 spilledH20flow[1,offpeak,Karapiro] = 0
 hydrojunctionconservation[1,Lake_Maraetai]:
   72000 turbineH20flow[1,peak,Maraetai]
   - 72000 turbineH20flow[1,peak,Whakamaru]
   + 216000 turbineH20flow[1,shoulder,Maraetai]
   - 216000 turbineH20flow[1,shoulder,Whakamaru]
   + 316800 turbineH20flow[1,offpeak,Maraetai]
   - 316800 turbineH20flow[1,offpeak,Whakamaru]
   + 72000 spilledH20flow[1,peak,Maraetai]
   - 72000 spilledH20flow[1,peak,Whakamaru]
   + 216000 spilledH20flow[1,shoulder,Maraetai]
   - 216000 spilledH20flow[1,shoulder,Whakamaru]
   + 316800 spilledH20flow[1,offpeak,Maraetai]
   - 316800 spilledH20flow[1,offpeak,Whakamaru] = 0
 hydrojunctionconservation[1,Lake_Ohakuri]:
   - 72000 turbineH20flow[1,peak,Aratiatia]
   + 72000 turbineH20flow[1,peak,Ohakuri]
   - 216000 turbineH20flow[1,shoulder,Aratiatia]
   + 216000 turbineH20flow[1,shoulder,Ohakuri]
   - 316800 turbineH20flow[1,offpeak,Aratiatia]
   + 316800 turbineH20flow[1,offpeak,Ohakuri]
   - 72000 spilledH20flow[1,peak,Aratiatia]
   + 72000 spilledH20flow[1,peak,Ohakuri]
   - 216000 spilledH20flow[1,shoulder,Aratiatia]
   + 216000 spilledH20flow[1,shoulder,Ohakuri]
   - 316800 spilledH20flow[1,offpeak,Aratiatia]
   + 316800 spilledH20flow[1,offpeak,Ohakuri] = 0
 hydrojunctionconservation[1,Lake_Waipapa]:
   - 72000 turbineH20flow[1,peak,Maraetai]
   + 72000 turbineH20flow[1,peak,Waipapa]
   - 216000 turbineH20flow[1,shoulder,Maraetai]
   + 216000 turbineH20flow[1,shoulder,Waipapa]
   - 316800 turbineH20flow[1,offpeak,Maraetai]
   + 316800 turbineH20flow[1,offpeak,Waipapa]
   - 72000 spilledH20flow[1,peak,Maraetai]
   + 72000 spilledH20flow[1,peak,Waipapa]
   - 216000 spilledH20flow[1,shoulder,Maraetai]
   + 216000 spilledH20flow[1,shoulder,Waipapa]
   - 316800 spilledH20flow[1,offpeak,Maraetai]
   + 316800 spilledH20flow[1,offpeak,Waipapa] = 0
 hydrojunctionconservation[1,Lake_Whakamaru]:
   - 72000 turbineH20flow[1,peak,Atiamuri]
   + 72000 turbineH20flow[1,peak,Whakamaru]
   - 216000 turbineH20flow[1,shoulder,Atiamuri]
   + 216000 turbineH20flow[1,shoulder,Whakamaru]
   - 316800 turbineH20flow[1,offpeak,Atiamuri]
   + 316800 turbineH20flow[1,offpeak,Whakamaru]
   - 72000 spilledH20flow[1,peak,Atiamuri]
   + 72000 spilledH20flow[1,peak,Whakamaru]
   - 216000 spilledH20flow[1,shoulder,Atiamuri]
   + 216000 spilledH20flow[1,shoulder,Whakamaru]
   - 316800 spilledH20flow[1,offpeak,Atiamuri]
   + 316800 spilledH20flow[1,offpeak,Whakamaru] = 0
 hydrojunctionconservation[1,Lake_Aviemore]:
   72000 turbineH20flow[1,peak,Aviemore]
   - 72000 turbineH20flow[1,peak,Benmore]
   + 216000 turbineH20flow[1,shoulder,Aviemore]
   - 216000 turbineH20flow[1,shoulder,Benmore]
   + 316800 turbineH20flow[1,offpeak,Aviemore]
   - 316800 turbineH20flow[1,offpeak,Benmore]
   + 72000 spilledH20flow[1,peak,Aviemore]
   - 72000 spilledH20flow[1,peak,Benmore]
   + 216000 spilledH20flow[1,shoulder,Aviemore]
   - 216000 spilledH20flow[1,shoulder,Benmore]
   + 316800 spilledH20flow[1,offpeak,Aviemore]
   - 316800 spilledH20flow[1,offpeak,Benmore] = 0
 hydrojunctionconservation[1,Lake_Waitaki]:
   - 72000 turbineH20flow[1,peak,Aviemore]
   + 72000 turbineH20flow[1,peak,Waitaki]
   - 216000 turbineH20flow[1,shoulder,Aviemore]
   + 216000 turbineH20flow[1,shoulder,Waitaki]
   - 316800 turbineH20flow[1,offpeak,Aviemore]
   + 316800 turbineH20flow[1,offpeak,Waitaki]
   - 72000 spilledH20flow[1,peak,Aviemore]
   + 72000 spilledH20flow[1,peak,Waitaki]
   - 216000 spilledH20flow[1,shoulder,Aviemore]
   + 216000 spilledH20flow[1,shoulder,Waitaki]
   - 316800 spilledH20flow[1,offpeak,Aviemore]
   + 316800 spilledH20flow[1,offpeak,Waitaki] = 0
 hydrojunctionconservation[1,Lake_Cobb]: 72000 turbineH20flow[1,peak,Cobb]
   + 216000 turbineH20flow[1,shoulder,Cobb]
   + 316800 turbineH20flow[1,offpeak,Cobb]
   + 72000 spilledH20flow[1,peak,Cobb]
   + 216000 spilledH20flow[1,shoulder,Cobb]
   + 316800 spilledH20flow[1,offpeak,Cobb] = 0
 hydrojunctionconservation[1,Lake_Coleridge]:
   72000 turbineH20flow[1,peak,Coleridge]
   + 216000 turbineH20flow[1,shoulder,Coleridge]
   + 316800 turbineH20flow[1,offpeak,Coleridge]
   + 72000 spilledH20flow[1,peak,Coleridge]
   + 216000 spilledH20flow[1,shoulder,Coleridge]
   + 316800 spilledH20flow[1,offpeak,Coleridge] = 0
 hydrojunctionconservation[1,Mangahao_head]:
   72000 turbineH20flow[1,peak,Mangahao]
   + 216000 turbineH20flow[1,shoulder,Mangahao]
   + 316800 turbineH20flow[1,offpeak,Mangahao]
   + 72000 spilledH20flow[1,peak,Mangahao]
   + 216000 spilledH20flow[1,shoulder,Mangahao]
   + 316800 spilledH20flow[1,offpeak,Mangahao] = 0
 hydrojunctionconservation[1,Lake_Matahina]:
   72000 turbineH20flow[1,peak,Matahina]
   + 216000 turbineH20flow[1,shoulder,Matahina]
   + 316800 turbineH20flow[1,offpeak,Matahina]
   + 72000 spilledH20flow[1,peak,Matahina]
   + 216000 spilledH20flow[1,shoulder,Matahina]
   + 316800 spilledH20flow[1,offpeak,Matahina] = 0
 hydrojunctionconservation[1,Lake_Moawhango]:
   72000 turbineH20flow[1,peak,Rangipo]
   + 216000 turbineH20flow[1,shoulder,Rangipo]
   + 316800 turbineH20flow[1,offpeak,Rangipo]
   + 72000 spilledH20flow[1,peak,Rangipo]
   + 216000 spilledH20flow[1,shoulder,Rangipo]
   + 316800 spilledH20flow[1,offpeak,Rangipo] = 0
 hydrojunctionconservation[1,Lake_Rotoaira]:
   72000 turbineH20flow[1,peak,Tokaanu]
   + 216000 turbineH20flow[1,shoulder,Tokaanu]
   + 316800 turbineH20flow[1,offpeak,Tokaanu]
   + 72000 spilledH20flow[1,peak,Tokaanu]
   + 216000 spilledH20flow[1,shoulder,Tokaanu]
   + 316800 spilledH20flow[1,offpeak,Tokaanu] = 0
 hydroreservoirservation[1,Lake_Benmore]:
   72000 turbineH20flow[1,peak,Benmore]
   - 72000 turbineH20flow[1,peak,Ohau_C]
   + 216000 turbineH20flow[1,shoulder,Benmore]
   - 216000 turbineH20flow[1,shoulder,Ohau_C]
   + 316800 turbineH20flow[1,offpeak,Benmore]
   - 316800 turbineH20flow[1,offpeak,Ohau_C]
   + 72000 spilledH20flow[1,peak,Benmore]
   - 72000 spilledH20flow[1,peak,Ohau_C]
   + 216000 spilledH20flow[1,shoulder,Benmore]
   - 216000 spilledH20flow[1,shoulder,Ohau_C]
   + 316800 spilledH20flow[1,offpeak,Benmore]
   - 316800 spilledH20flow[1,offpeak,Ohau_C]
   - 72000 arcflow[1,peak,Lake_Scott,Lake_Benmore]
   - 72000 arcflow[1,peak,Lake_Pukaki,Lake_Benmore]
   - 72000 arcflow[1,peak,Lake_Ruataniwha,Lake_Benmore]
   - 72000 arcflow[1,peak,Ohau_BC_canal,Lake_Benmore]
   - 216000 arcflow[1,shoulder,Lake_Scott,Lake_Benmore]
   - 216000 arcflow[1,shoulder,Lake_Pukaki,Lake_Benmore]
   - 216000 arcflow[1,shoulder,Lake_Ruataniwha,Lake_Benmore]
   - 216000 arcflow[1,shoulder,Ohau_BC_canal,Lake_Benmore]
   - 316800 arcflow[1,offpeak,Lake_Scott,Lake_Benmore]
   - 316800 arcflow[1,offpeak,Lake_Pukaki,Lake_Benmore]
   - 316800 arcflow[1,offpeak,Lake_Ruataniwha,Lake_Benmore]
   - 316800 arcflow[1,offpeak,Ohau_BC_canal,Lake_Benmore]
   + endstorage[1,Lake_Benmore] - startstorage[1,Lake_Benmore] = 0
 hydroreservoirservation[1,Lake_Hawea]:
   72000 arcflow[1,peak,Lake_Hawea,Lake_Dunstan]
   + 216000 arcflow[1,shoulder,Lake_Hawea,Lake_Dunstan]
   + 316800 arcflow[1,offpeak,Lake_Hawea,Lake_Dunstan]
   + endstorage[1,Lake_Hawea] - startstorage[1,Lake_Hawea] = 0
 hydroreservoirservation[1,Lake_Ohau]:
   72000 arcflow[1,peak,Lake_Ohau,Pukaki_Ohau_canal_junction]
   + 72000 arcflow[1,peak,Lake_Ohau,Lake_Ruataniwha]
   + 216000 arcflow[1,shoulder,Lake_Ohau,Pukaki_Ohau_canal_junction]
   + 216000 arcflow[1,shoulder,Lake_Ohau,Lake_Ruataniwha]
   + 316800 arcflow[1,offpeak,Lake_Ohau,Pukaki_Ohau_canal_junction]
   + 316800 arcflow[1,offpeak,Lake_Ohau,Lake_Ruataniwha]
   + endstorage[1,Lake_Ohau] - startstorage[1,Lake_Ohau] = 0
 hydroreservoirservation[1,Lake_Pukaki]:
   - 72000 turbineH20flow[1,peak,Tekapo_B]
   - 216000 turbineH20flow[1,shoulder,Tekapo_B]
   - 316800 turbineH20flow[1,offpeak,Tekapo_B]
   - 72000 spilledH20flow[1,peak,Tekapo_B]
   - 216000 spilledH20flow[1,shoulder,Tekapo_B]
   - 316800 spilledH20flow[1,offpeak,Tekapo_B]
   + 72000 arcflow[1,peak,Lake_Pukaki,Lake_Benmore]
   + 72000 arcflow[1,peak,Lake_Pukaki,Pukaki_Ohau_canal_junction]
   + 216000 arcflow[1,shoulder,Lake_Pukaki,Lake_Benmore]
   + 216000 arcflow[1,shoulder,Lake_Pukaki,Pukaki_Ohau_canal_junction]
   + 316800 arcflow[1,offpeak,Lake_Pukaki,Lake_Benmore]
   + 316800 arcflow[1,offpeak,Lake_Pukaki,Pukaki_Ohau_canal_junction]
   + endstorage[1,Lake_Pukaki] - startstorage[1,Lake_Pukaki] = 0
 hydroreservoirservation[1,Lake_Taupo]:
   72000 arcflow[1,peak,Lake_Taupo,Lake_Aratiatia]
   + 216000 arcflow[1,shoulder,Lake_Taupo,Lake_Aratiatia]
   + 316800 arcflow[1,offpeak,Lake_Taupo,Lake_Aratiatia]
   + endstorage[1,Lake_Taupo] - startstorage[1,Lake_Taupo] = 0
 hydroreservoirservation[1,Lake_Tekapo]:
   72000 turbineH20flow[1,peak,Tekapo_A]
   + 216000 turbineH20flow[1,shoulder,Tekapo_A]
   + 316800 turbineH20flow[1,offpeak,Tekapo_A]
   + 72000 spilledH20flow[1,peak,Tekapo_A]
   + 216000 spilledH20flow[1,shoulder,Tekapo_A]
   + 316800 spilledH20flow[1,offpeak,Tekapo_A]
   + 72000 arcflow[1,peak,Lake_Tekapo,Lake_Scott]
   + 216000 arcflow[1,shoulder,Lake_Tekapo,Lake_Scott]
   + 316800 arcflow[1,offpeak,Lake_Tekapo,Lake_Scott]
   + endstorage[1,Lake_Tekapo] - startstorage[1,Lake_Tekapo] = 0
 hydroreservoirservation[1,Lakes_Manapouri_Te_Anau]:
   72000 turbineH20flow[1,peak,Manapouri]
   + 216000 turbineH20flow[1,shoulder,Manapouri]
   + 316800 turbineH20flow[1,offpeak,Manapouri]
   + 72000 spilledH20flow[1,peak,Manapouri]
   + 216000 spilledH20flow[1,shoulder,Manapouri]
   + 316800 spilledH20flow[1,offpeak,Manapouri]
   + endstorage[1,Lakes_Manapouri_Te_Anau]
   - startstorage[1,Lakes_Manapouri_Te_Anau] = 0
 hydroreservoirservation[1,Lake_Waikaremoana]:
   72000 turbineH20flow[1,peak,Waikaremoana]
   + 216000 turbineH20flow[1,shoulder,Waikaremoana]
   + 316800 turbineH20flow[1,offpeak,Waikaremoana]
   + 72000 spilledH20flow[1,peak,Waikaremoana]
   + 216000 spilledH20flow[1,shoulder,Waikaremoana]
   + 316800 spilledH20flow[1,offpeak,Waikaremoana]
   + endstorage[1,Lake_Waikaremoana] - startstorage[1,Lake_Waikaremoana]
   = 0
 startstoragegwh[Lake_Benmore]: startstorage[1,Lake_Benmore] = 0
 startstoragegwh[Lake_Hawea]: startstorage[1,Lake_Hawea] = 1.0922591e+09
 startstoragegwh[Lake_Ohau]: startstorage[1,Lake_Ohau] = 0
 startstoragegwh[Lake_Pukaki]: startstorage[1,Lake_Pukaki] = 2.0591525e+09
 startstoragegwh[Lake_Taupo]: startstorage[1,Lake_Taupo] = 6.408265e+08
 startstoragegwh[Lake_Tekapo]: startstorage[1,Lake_Tekapo] = 5.530834e+08
 startstoragegwh[Lakes_Manapouri_Te_Anau]:
   startstorage[1,Lakes_Manapouri_Te_Anau] = 8.186229e+08
 startstoragegwh[Lake_Waikaremoana]: startstorage[1,Lake_Waikaremoana] = 0
Bounds
 generation[1,peak,Huntly_U6] <= 43.2
 generation[1,peak,Huntly_U5] <= 346.5
 generation[1,peak,TCC] <= 339.3
 generation[1,peak,NI_Thermal] <= 540
 generation[1,peak,SI_Thermal] = 0
 generation[1,peak,Arapuni] <= 192
 generation[1,peak,Aratiatia] <= 78
 generation[1,peak,Atiamuri] <= 84
 generation[1,peak,Karapiro] <= 96
 generation[1,peak,Maraetai] <= 352
 generation[1,peak,Ohakuri] <= 112
 generation[1,peak,Waipapa] <= 54
 generation[1,peak,Whakamaru] <= 110
 generation[1,peak,Aviemore] <= 220
 generation[1,peak,Benmore] <= 540
 generation[1,peak,Ohau_A] <= 264
 generation[1,peak,Ohau_B] <= 212
 generation[1,peak,Ohau_C] <= 212
 generation[1,peak,Tekapo_B] <= 154
 generation[1,peak,Tekapo_A] <= 27
 generation[1,peak,Waitaki] <= 105
 generation[1,peak,Clyde_220kV] <= 432
 generation[1,peak,Roxburgh] <= 320
 generation[1,peak,Manapouri] <= 800
 generation[1,peak,Cobb] <= 32
 generation[1,peak,Coleridge] <= 39
 generation[1,peak,Mangahao] <= 30
 generation[1,peak,Matahina] <= 80
 generation[1,peak,Rangipo] <= 120
 generation[1,peak,Tokaanu] <= 240
 generation[1,peak,Waikaremoana] <= 140
 generation[1,shoulder,Huntly_U6] <= 43.2
 generation[1,shoulder,Huntly_U5] <= 346.5
 generation[1,shoulder,TCC] <= 339.3
 generation[1,shoulder,NI_Thermal] <= 540
 generation[1,shoulder,SI_Thermal] = 0
 generation[1,shoulder,Arapuni] <= 192
 generation[1,shoulder,Aratiatia] <= 78
 generation[1,shoulder,Atiamuri] <= 84
 generation[1,shoulder,Karapiro] <= 96
 generation[1,shoulder,Maraetai] <= 352
 generation[1,shoulder,Ohakuri] <= 112
 generation[1,shoulder,Waipapa] <= 54
 generation[1,shoulder,Whakamaru] <= 110
 generation[1,shoulder,Aviemore] <= 220
 generation[1,shoulder,Benmore] <= 540
 generation[1,shoulder,Ohau_A] <= 264
 generation[1,shoulder,Ohau_B] <= 212
 generation[1,shoulder,Ohau_C] <= 212
 generation[1,shoulder,Tekapo_B] <= 154
 generation[1,shoulder,Tekapo_A] <= 27
 generation[1,shoulder,Waitaki] <= 105
 generation[1,shoulder,Clyde_220kV] <= 432
 generation[1,shoulder,Roxburgh] <= 320
 generation[1,shoulder,Manapouri] <= 800
 generation[1,shoulder,Cobb] <= 32
 generation[1,shoulder,Coleridge] <= 39
 generation[1,shoulder,Mangahao] <= 30
 generation[1,shoulder,Matahina] <= 80
 generation[1,shoulder,Rangipo] <= 120
 generation[1,shoulder,Tokaanu] <= 240
 generation[1,shoulder,Waikaremoana] <= 140
 generation[1,offpeak,Huntly_U6] <= 43.2
 generation[1,offpeak,Huntly_U5] <= 346.5
 generation[1,offpeak,TCC] <= 339.3
 generation[1,offpeak,NI_Thermal] <= 540
 generation[1,offpeak,SI_Thermal] = 0
 generation[1,offpeak,Arapuni] <= 192
 generation[1,offpeak,Aratiatia] <= 78
 generation[1,offpeak,Atiamuri] <= 84
 generation[1,offpeak,Karapiro] <= 96
 generation[1,offpeak,Maraetai] <= 352
 generation[1,offpeak,Ohakuri] <= 112
 generation[1,offpeak,Waipapa] <= 54
 generation[1,offpeak,Whakamaru] <= 110
 generation[1,offpeak,Aviemore] <= 220
 generation[1,offpeak,Benmore] <= 540
 generation[1,offpeak,Ohau_A] <= 264
 generation[1,offpeak,Ohau_B] <= 212
 generation[1,offpeak,Ohau_C] <= 212
 generation[1,offpeak,Tekapo_B] <= 154
 generation[1,offpeak,Tekapo_A] <= 27
 generation[1,offpeak,Waitaki] <= 105
 generation[1,offpeak,Clyde_220kV] <= 432
 generation[1,offpeak,Roxburgh] <= 320
 generation[1,offpeak,Manapouri] <= 800
 generation[1,offpeak,Cobb] <= 32
 generation[1,offpeak,Coleridge] <= 39
 generation[1,offpeak,Mangahao] <= 30
 generation[1,offpeak,Matahina] <= 80
 generation[1,offpeak,Rangipo] <= 120
 generation[1,offpeak,Tokaanu] <= 240
 generation[1,offpeak,Waikaremoana] <= 140
 shortage[1,peak,HY,ls1] <= 8.54125
 shortage[1,peak,HY,ls2] <= 8.54125
 shortage[1,peak,HY,ls3] <= 17.0825
 shortage[1,peak,HY,ls4] <= 51.2475
 shortage[1,peak,HY,ls5] <= 256.2375
 shortage[1,peak,NI,ls1] <= 68.0425
 shortage[1,peak,NI,ls2] <= 68.0425
 shortage[1,peak,NI,ls3] <= 136.085
 shortage[1,peak,NI,ls4] <= 408.255
 shortage[1,peak,NI,ls5] <= 2041.275
 shortage[1,peak,SI,ls1] <= 46.5925
 shortage[1,peak,SI,ls2] <= 46.5925
 shortage[1,peak,SI,ls3] <= 93.185
 shortage[1,peak,SI,ls4] <= 279.555
 shortage[1,peak,SI,ls5] <= 1397.775
 shortage[1,shoulder,HY,ls1] <= 7.782075000000001
 shortage[1,shoulder,HY,ls2] <= 7.782075000000001
 shortage[1,shoulder,HY,ls3] <= 15.56415
 shortage[1,shoulder,HY,ls4] <= 46.69245
 shortage[1,shoulder,HY,ls5] <= 233.46225
 shortage[1,shoulder,NI,ls1] <= 63.966675
 shortage[1,shoulder,NI,ls2] <= 63.966675
 shortage[1,shoulder,NI,ls3] <= 127.93335
 shortage[1,shoulder,NI,ls4] <= 383.80005
 shortage[1,shoulder,NI,ls5] <= 1919.00025
 shortage[1,shoulder,SI,ls1] <= 44.525
 shortage[1,shoulder,SI,ls2] <= 44.525
 shortage[1,shoulder,SI,ls3] <= 89.05
 shortage[1,shoulder,SI,ls4] <= 267.15
 shortage[1,shoulder,SI,ls5] <= 1335.75
 shortage[1,offpeak,HY,ls1] <= 5.773
 shortage[1,offpeak,HY,ls2] <= 5.773
 shortage[1,offpeak,HY,ls3] <= 11.546
 shortage[1,offpeak,HY,ls4] <= 34.638
 shortage[1,offpeak,HY,ls5] <= 173.19
 shortage[1,offpeak,NI,ls1] <= 50.268475
 shortage[1,offpeak,NI,ls2] <= 50.268475
 shortage[1,offpeak,NI,ls3] <= 100.53695
 shortage[1,offpeak,NI,ls4] <= 301.61085
 shortage[1,offpeak,NI,ls5] <= 1508.05425
 shortage[1,offpeak,SI,ls1] <= 39.85482500000001
 shortage[1,offpeak,SI,ls2] <= 39.85482500000001
 shortage[1,offpeak,SI,ls3] <= 79.70965000000001
 shortage[1,offpeak,SI,ls4] <= 239.12895
 shortage[1,offpeak,SI,ls5] <= 1195.64475
 powerflow[1,peak,NI,HY] <= 1000
 powerflow[1,peak,HY,NI] <= 1000
 powerflow[1,peak,HY,SI] <= 900
 powerflow[1,peak,SI,HY] <= 900
 powerflow[1,shoulder,NI,HY] <= 1000
 powerflow[1,shoulder,HY,NI] <= 1000
 powerflow[1,shoulder,HY,SI] <= 900
 powerflow[1,shoulder,SI,HY] <= 900
 powerflow[1,offpeak,NI,HY] <= 1000
 powerflow[1,offpeak,HY,NI] <= 1000
 powerflow[1,offpeak,HY,SI] <= 900
 powerflow[1,offpeak,SI,HY] <= 900
 spilledH20flow[1,peak,Aviemore] <= 5400
 spilledH20flow[1,peak,Benmore] <= 3400
 spilledH20flow[1,peak,Ohau_A] = 0
 spilledH20flow[1,peak,Ohau_B] = 0
 spilledH20flow[1,peak,Ohau_C] <= 560
 spilledH20flow[1,peak,Tekapo_B] = 0
 spilledH20flow[1,peak,Tekapo_A] = 0
 spilledH20flow[1,peak,Waitaki] <= 5380
 spilledH20flow[1,peak,Clyde_220kV] <= 4140
 spilledH20flow[1,peak,Roxburgh] <= 6000
 spilledH20flow[1,shoulder,Aviemore] <= 5400
 spilledH20flow[1,shoulder,Benmore] <= 3400
 spilledH20flow[1,shoulder,Ohau_A] = 0
 spilledH20flow[1,shoulder,Ohau_B] = 0
 spilledH20flow[1,shoulder,Ohau_C] <= 560
 spilledH20flow[1,shoulder,Tekapo_B] = 0
 spilledH20flow[1,shoulder,Tekapo_A] = 0
 spilledH20flow[1,shoulder,Waitaki] <= 5380
 spilledH20flow[1,shoulder,Clyde_220kV] <= 4140
 spilledH20flow[1,shoulder,Roxburgh] <= 6000
 spilledH20flow[1,offpeak,Aviemore] <= 5400
 spilledH20flow[1,offpeak,Benmore] <= 3400
 spilledH20flow[1,offpeak,Ohau_A] = 0
 spilledH20flow[1,offpeak,Ohau_B] = 0
 spilledH20flow[1,offpeak,Ohau_C] <= 560
 spilledH20flow[1,offpeak,Tekapo_B] = 0
 spilledH20flow[1,offpeak,Tekapo_A] = 0
 spilledH20flow[1,offpeak,Waitaki] <= 5380
 spilledH20flow[1,offpeak,Clyde_220kV] <= 4140
 spilledH20flow[1,offpeak,Roxburgh] <= 6000
 endstorage[1,Lake_Benmore] = 0
 endstorage[1,Lake_Hawea] <= 1.40582e+09
 endstorage[1,Lake_Ohau] = 0
 endstorage[1,Lake_Pukaki] <= 2.67021e+09
 endstorage[1,Lake_Taupo] <= 8.554e+08
 endstorage[1,Lake_Tekapo] <= 7.44433333e+08
 endstorage[1,Lakes_Manapouri_Te_Anau] <= 1.02923e+09
 endstorage[1,Lake_Waikaremoana] = 0
 startstorage[1,Lake_Benmore] = 0
 startstorage[1,Lake_Hawea] <= 1.40582e+09
 startstorage[1,Lake_Ohau] = 0
 startstorage[1,Lake_Pukaki] <= 2.67021e+09
 startstorage[1,Lake_Taupo] <= 8.554e+08
 startstorage[1,Lake_Tekapo] <= 7.44433333e+08
 startstorage[1,Lakes_Manapouri_Te_Anau] <= 1.02923e+09
 startstorage[1,Lake_Waikaremoana] = 0
 endstorage_gwh[vs1] <= 992.98
 endstorage_gwh[vs2] <= 512.82
 endstorage_gwh[vs3] <= 201.48
 endstorage_gwh[vs4] <= 366.04
 endstorage_gwh[vs5] <= 244.39
 endstorage_gwh[vs6] <= 305.28
 endstorage_gwh[vs7] <= 66.06
 endstorage_gwh[vs8] <= 144.42
 endstorage_gwh[vs9] <= 89.05
 endstorage_gwh[vs10] <= 177.87
 endstorage_gwh[vs11] <= 84.7
 endstorage_gwh[vs12] <= 279.89
 endstorage_gwh[vs13] <= 44.98
 endstorage_gwh[vs14] <= 114.89
 endstorage_gwh[vs15] <= 253.41
 endstorage_gwh[vs16] <= 308.32
End
