# -*- coding: utf-8 -*-
"""
Created on Thu Jul 05 17:18:21 2018

@author: nguyent
"""
import pandas as pd
import numpy as np
import gurobipy as grb


"""
Geting data from csv files and prepare inputs for the model
"""

csvinput  = r'C:\Security Margin\Development\Gurobi\CSV'

hydroarc  = pd.read_csv(csvinput + '\\hydroarc.csv',
                        comment = '%', sep = ',', header = 'infer')

reservoir  = pd.read_csv(csvinput + '\\reservoir.csv',
                         comment = '%', sep = ',', header = 'infer')

hydroplant = pd.read_csv(csvinput + '\\hydrostation.csv',
                         comment = '%', sep = ',', header = 'infer')

thermalplant = pd.read_csv(csvinput + '\\thermalstation.csv',
                           comment = '%', sep = ',', header = 'infer')

fuelcost = pd.read_csv(csvinput + '\\fuelcost.csv',
                       comment = '%', sep = ',', header = 'infer')

co2cost = pd.read_csv(csvinput + '\\co2cost.csv',
                      comment = '%', sep = ',', header = 'infer')

emission = pd.read_csv(csvinput + '\\emission.csv',
                       comment = '%', sep = ',', header = 'infer')

transmission = pd.read_csv(csvinput+ '\\transmission.csv',
                           comment = '%', sep = ',', header = 'infer')

demand  = pd.read_csv(csvinput + '\\demand.csv',
                      comment = '%', sep = ',', header = 'infer')

fixedstation = pd.read_csv(csvinput + '\\fixedstation.csv',
                           comment = '%', sep = ',', header = 'infer')

inflow = pd.read_csv(csvinput+'\inflow.csv',
                      comment = '%', sep = ',', header = 'infer')

sampledinflow = pd.read_csv(csvinput+ '\\DOASA_sample_inflow_400.csv',
                             comment = '%', sep = ',', header = 'infer')

blockhour  = pd.read_csv(csvinput + '\\blockhour.csv',
                         comment = '%', sep = ',', header = 'infer')

voll = pd.read_csv(csvinput + '\\lossloadsegment.csv',
                   comment = '%', sep = ',', header = 'infer')

endh2ovalue = pd.read_csv(csvinput + '\\terminalwatervalue.csv',
                          comment = '%', sep = ',', header = 'infer')


"""
End of geting data from csv files and prepare inputs for the model
"""



# Creating scenario, week, block and node series
scenario = sampledinflow['scenario'].drop_duplicates().tolist()
scenario.sort()

week = demand['week'].drop_duplicates().tolist()
week.sort()
weekr = list(week)
weekr.remove(1)
weekr.reverse()

block = list(blockhour.columns.values)
if 'week'in block: block.remove('week')

node = list(demand.columns.values)
if 'week' in node: node.remove('week')
if 'block' in node: node.remove('block')


# Creating generator lists and generatornode tuplelist
df = thermalplant[['generator','node','capacity']] \
    .append(hydroplant[['generator','node','capacity']]) \
    .drop_duplicates().set_index(['generator'])
df.sort_index()
generator = df.index.tolist()
generatorthermal = thermalplant['generator'].tolist()
generatorhydro = hydroplant['generator'].tolist()
generator_capacity = dict(df['capacity'])
df = df.reset_index().set_index(['generator','node'])
generatornode = grb.tuplelist(df.index.tolist())


# Mapping hydro plants behind a reservoir
a = hydroarc[['frjunc','tojunc']].append(hydroplant[['frjunc','tojunc']])
a = a.drop_duplicates()
b = a[a['frjunc'].isin(reservoir['reservoir'])]
b = b.rename(index=str, columns={"frjunc": "reservoir"})
c = pd.DataFrame({'reservoir':reservoir['reservoir'],
                  'frjunc':reservoir['reservoir']})
while b.shape[0] > 0:
    b['frjunc'] = b['tojunc']
    b = b[['reservoir','frjunc']]
    c = c.append(b[['reservoir','frjunc']])
    b = pd.merge(b, a, on='frjunc')

# Calculate reservoir power factor
c = pd.merge(c,hydroplant, on = 'frjunc').drop_duplicates()
reservoirfactor = c.groupby(['reservoir'])['powerfactor'].sum()
reservoirfactor = dict(reservoirfactor)
del(a,b,c)


# Creating list of reservoir and reservoir dictionaries
df = reservoir.set_index(['reservoir'])
reservoir_capacity = dict(df['capacity'])
reservoir_contingent = dict(df['contingent'])
reservoir_initial = dict(df['initial'])
reservoir = df.index.tolist()


# Creating list of hydro junctions including reservoirs
junction = hydroarc['frjunc'].append(hydroarc['tojunc']).    \
           append(hydroplant['frjunc']).append(hydroplant['tojunc']). \
           unique().tolist()


# Getting list of hydro arc and minflow and maxflow dictionaries
arc = hydroarc.set_index(['frjunc','tojunc'])
arc['minflow'] = arc['minflow'].apply(lambda x: 0 if x.strip() =='na' \
                                                  else float(x.strip()))
arc['maxflow'] = arc['maxflow'].apply(lambda x: 1e6 if x.strip() =='na' \
                                                    else float(x.strip()))
minflow = arc[arc['minflow'] > 0]
minflowarc = grb.tuplelist(minflow.index.tolist())
minflow = dict(minflow['minflow'])
maxflow = arc[arc['maxflow'] < 1e6]
maxflowarc = grb.tuplelist(maxflow.index.tolist())
maxflow = dict(maxflow['maxflow'])
arc = grb.tuplelist(arc.index.tolist())


# Getting hydro plant data
hydroplant['maxspill'] = hydroplant['maxspill'] \
                        .apply(lambda x: 1e6 if x.strip()=='na' \
                                             else float(x.strip()))
hydroplant_maxspill = hydroplant[hydroplant['maxspill'] < 1e6]
hydroplant_maxspill = hydroplant_maxspill.set_index(['generator'])
hydroplant_maxspill = dict(hydroplant_maxspill['maxspill'])

hydroplant_powrfact = hydroplant.set_index(['generator'])
hydroplant_powrfact = dict(hydroplant_powrfact['powerfactor'])

hydroplantfr = hydroplant.set_index(['generator','frjunc'])
hydroplantfr = grb.tuplelist(hydroplantfr.index.tolist())
hydroplantto = hydroplant.set_index(['generator','tojunc'])
hydroplantto = grb.tuplelist(hydroplantto.index.tolist())


# Calculate fuel cost adjsuted for emission cost
df = pd.melt(fuelcost, id_vars=['week'],value_vars = ['coal','diesel','gas'],
             var_name = 'fuel', value_name = 'dollarpergj')
df = pd.merge(df, emission, on = 'fuel')
df = pd.merge(df, co2cost, on = 'week')
df['co2costpergj'] = df['tonperpj'] * df['nzdperton'] / 1e6
df['dollarpergj'] = df['dollarpergj'] + df['co2costpergj']
fuelco2cost = df[['week','fuel','dollarpergj']]


# Calculate thermal plant weekly srmc
srmc = thermalplant[['generator','fuel','heatrate','operationcost']]
srmc = pd.merge(srmc,fuelco2cost, on = 'fuel')
srmc['dollarpermwh'] = srmc['heatrate'] * srmc['dollarpergj'] \
                     + srmc['operationcost']
srmc = srmc[['week','generator','dollarpermwh']]
srmc = srmc.set_index(['week','generator'])
srmc = dict(srmc['dollarpermwh'])


# Create transmission list ad dictionaries data
tx = transmission.set_index(['frnode','tonode'])
tx_capacity = dict(tx['capacity'])
tx = grb.tuplelist(tx.index.tolist())


# Calculate node net demand
dm = pd.melt(demand,id_vars = ['week','block'], value_vars = ['HY','NI','SI'],
             var_name = 'node', value_name = 'demand')
fx = pd.melt(fixedstation,id_vars = ['generator','node','week'],
             value_vars = ['peak','shoulder','offpeak'],
             var_name = 'block',value_name = 'fixedgen') \
             .groupby(['week','block','node'], as_index = False).sum()
dm = pd.merge(dm,fx,how='left', on = ['week','block','node'])
dm['fixedgen'] = dm['fixedgen'].apply(lambda x: 0 if np.isnan(x) else x)
dm['MW'] = dm['demand'] - dm['fixedgen']
dm = dm.set_index(['week','block','node'])
demandmw = dict(dm['demand'])
netdemand = dict(dm['MW'])
del(dm,fx)


# Get inflow data
historicalinflow = pd.melt(inflow,id_vars = ['scenario','week'],
                           var_name = 'lake',value_name = 'inflow')

sampledinflow = pd.melt(sampledinflow,id_vars = ['scenario','week'],
                        var_name = 'lake',value_name = 'inflow')

# Get block hour data
blockhour = pd.melt(blockhour,id_vars = ['week'],
                    var_name = 'block',value_name = 'hour')

blkhour = blockhour.set_index(['week','block'])
blkhour = dict(blkhour['hour'])

# Get shortage penalty data
vollsegment = voll.segment.drop_duplicates().tolist()
vollsegment.sort()
df = voll.set_index(['node','segment'])
shortageprop = dict(df['proportion'])
shortagecost = dict(df['cost'])
del(df)



# Calculate end storage value piecwise curve
endh2ovalue['lb'] = endh2ovalue.gwh.shift(1)
endh2ovalue.at[0,'lb'] = 0
endh2ovalue['gwh'] = endh2ovalue['gwh'] - endh2ovalue['lb']
endh2ovalue.drop(['lb'], axis = 1, inplace = True)
endh2ovalue = endh2ovalue.set_index('segment')
h2osegment = endh2ovalue.index.tolist()
h2osegmentgwh = dict(endh2ovalue['gwh'])
h2odlrpermwh = dict(endh2ovalue['dollarpermwh'])



maxflowcost = 500
minflowcost = 500


slopes = pd.DataFrame( columns = ['week','reservoir','iter','value'])
intercepts = pd.DataFrame( columns = ['week','iter','value'])


for s in range(1,11):

    endstorage = reservoir_initial
    storedstartstorage = dict()

    """ FORWARD SOLVE """
    for w in week:

        for r in reservoir:
            storedstartstorage.update({(w,r): endstorage[r]})

        #  Set model parameters
        inflow = sampledinflow[ (sampledinflow['scenario'] == s) \
                              & (sampledinflow['week'] == w)]
        inflow = inflow.set_index(['lake'])
        inflow = dict(inflow['inflow'])

        startstorage = endstorage
        
        if w < week[-1]:
            slope = slopes[slopes['week']==w].set_index(['reservoir','iter'])
            slope = dict(slope['value'])

            intercept = intercepts[intercepts['week']==w].set_index(['iter'])
            intercept = dict(intercept['value'])

            iteration = intercepts[intercepts['week']==w]['iter'].tolist()

        
        # Create optimization model
        m = grb.Model('HydroScheduling')


        # Create variables ####################################################
        
        upperbound = [ generator_capacity[g] 
                       for b in block for g in generator ]
        GENERATION = m.addVars(block, generator, 
                               ub=upperbound, name='generation')

        upperbound = [ shortageprop[n,vs] * demandmw[w,b,n] 
                       for b in block for n in node for vs in vollsegment ]
        SHORTAGE_MW = m.addVars(block,node,vollsegment,
                                ub = upperbound, name='shortage_mw')

        upperbound = [tx_capacity[br] for b in block for br in tx ]
        POWERFLOW = m.addVars(block, tx, ub = upperbound, name='powerflow')

        TURBINEH2OFLOW = m.addVars(block, generatorhydro, 
                                   name = 'turbineh2oflow')

        upperbound = [ hydroplant_maxspill[g] if g in hydroplant_maxspill 
                       else grb.GRB.INFINITY 
                       for b in block for g in generatorhydro]
        SPILLEDH2OFLOW = m.addVars(block, generatorhydro,
                                   ub=upperbound, name='spilledh2oflow')

        FUTURECOST = m.addVar(name = 'futurecost')

        ARCFLOW = m.addVars(block, arc, name='arcflow')

        ARCFLOWDEFICIT = m.addVars(block, maxflowarc, name='arcflowdeficit')

        ARCFLOWSURPLUS = m.addVars(block, minflowarc, name='arcflowsurplus')

        upperbound = [reservoir_capacity[r] for r in reservoir]
        ENDSTORAGE = m.addVars(reservoir, ub = upperbound, name='endstorage')

        if w == week[-1]:
            ENDSTORAGE_GWH = m.addVars(h2osegment, name='endstorage_gwh')
        # Create variables end ################################################
            
            

        # Create objective function ###########################################
        if w < week[-1]:
            m.setObjective(
                  grb.quicksum( GENERATION[b,g]
                                * srmc[w,g] * blkhour[w,b]
                                for b in block 
                                for g in generatorthermal )
                + grb.quicksum( SHORTAGE_MW[b,n,ls] 
                                * shortagecost[n,ls] * blkhour[w,b] 
                                for b in block 
                                for n in node 
                                for ls in vollsegment )
                + grb.quicksum( ARCFLOWDEFICIT[b,i,j]
                                * maxflowcost * blkhour[w,b]
                                for b in block 
                                for (i,j) in maxflowarc )
                + grb.quicksum( ARCFLOWSURPLUS[b,i,j] 
                                * minflowcost * blkhour[w,b]
                                for b in block 
                                for (i,j) in minflowarc)
                + FUTURECOST
            )

        if w == week[-1]:
            m.setObjective(
              grb.quicksum( GENERATION[b,g] 
                            * srmc[w,g] * blkhour[w,b]
                            for b in block 
                            for g in generatorthermal )
            + grb.quicksum( SHORTAGE_MW[b,n,ls]
                            * shortagecost[n,ls] * blkhour[w,b] 
                            for b in block 
                            for n in node 
                            for ls in vollsegment )
            + grb.quicksum( ARCFLOWDEFICIT[b,i,j] 
                            * maxflowcost * blkhour[w,b]
                            for b in block 
                            for (i,j) in maxflowarc )
            + grb.quicksum( ARCFLOWSURPLUS[b,i,j] 
                            * minflowcost * blkhour[w,b]
                            for b in block 
                            for (i,j) in minflowarc)
            - grb.quicksum( ENDSTORAGE_GWH[vs] 
                            * h2odlrpermwh[vs] * 1e3
                            for vs in h2osegment )
            )
        # Create objective function end #######################################
            
            
        
        # Create constraints ##################################################
        
        # Future cost constraints
        if w < week[-1]:
            m.addConstrs(
                ( FUTURECOST >= intercept[i]
                              + grb.quicksum( ENDSTORAGE[r] * slope[r,i]
                                              for r in reservoir )
                 for i in iteration
                ),  name = "futurecost")

        # Supply demand balance constrant
        m.addConstrs(
            (   grb.quicksum( GENERATION[b,g]
                              for g,n in generatornode.select('*',n))
              + grb.quicksum( POWERFLOW[b,fr,n]
                              for fr,n in tx.select('*',n))
              - grb.quicksum( POWERFLOW[b,n,to]
                              for n,to in tx.select(n,'*'))
              + grb.quicksum( SHORTAGE_MW[b,n,ls]
                              for ls in vollsegment)
            ==  
                netdemand[w,b,n]
                for b in block 
                for n in node
            ),  name = 'supplydemandbalance')

        # Hydro generation conversion
        m.addConstrs(
            (   GENERATION[b,g] 
            == 
                TURBINEH2OFLOW[b,g] * hydroplant_powrfact[g]
                for b in block 
                for g in generatorhydro
            ),  name = 'hydrogenerationconversion')


        # Maximum hydro flow constraint
        m.addConstrs(
            (   ARCFLOW[b,i,j] - ARCFLOWDEFICIT[b,i,j]
            <=  
                maxflow[i,j]
                for b in block 
                for (i,j) in maxflowarc
            ),  name = 'maxhydroflow')

        # Minimum hydro flow constraint
        m.addConstrs(
            (   ARCFLOW[b,i,j] + ARCFLOWSURPLUS[b,i,j]
            >=  
                minflow[i,j]
                for b in block 
                for (i,j) in minflowarc
            ),  name = 'minhydroflow')

        # Hydro junction conservation constraint
        m.addConstrs(
            (   grb.quicksum( ARCFLOW.sum(b,j,'*') * blkhour[w,b] * 3600
                            + grb.quicksum( TURBINEH2OFLOW[b,g1] 
                                            * blkhour[w,b] * 3600
                                          + SPILLEDH2OFLOW[b,g1] 
                                            * blkhour[w,b] * 3600
                                            for g1 in generatorhydro
                                            if (g1,j) in hydroplantfr )
                              for b in block )
              - grb.quicksum( ARCFLOW.sum(b,'*',j) * blkhour[w,b] * 3600
                            + grb.quicksum( TURBINEH2OFLOW[b,g2] 
                                            * blkhour[w,b] * 3600
                                          + SPILLEDH2OFLOW[b,g2] 
                                            * blkhour[w,b] * 3600
                                            for g2 in generatorhydro
                                            if (g2,j) in hydroplantto )
                              for b in block )
            ==
                grb.quicksum( inflow[j] * blkhour[w,b] * 3600
                              for b in block 
                              if j in inflow )

                for j in junction 
                if j <> 'SEA' and j not in reservoir

            ),  name = 'hydrojunctionconservation')
        
         # Hydro reservoir conservation constraint
        m.addConstrs(
            (   grb.quicksum( ARCFLOW.sum(b,r,'*') * blkhour[w,b] * 3600
                            + grb.quicksum( TURBINEH2OFLOW[b,g1] 
                                            * blkhour[w,b] * 3600
                                          + SPILLEDH2OFLOW[b,g1] 
                                            * blkhour[w,b] * 3600
                                            for g1 in generatorhydro
                                            if (g1,r) in hydroplantfr )
                              for b in block )
              - grb.quicksum( ARCFLOW.sum(b,'*',r) * blkhour[w,b] * 3600
                            + grb.quicksum( TURBINEH2OFLOW[b,g2] 
                                            * blkhour[w,b] * 3600
                                          + SPILLEDH2OFLOW[b,g2] 
                                            * blkhour[w,b] * 3600
                                            for g2 in generatorhydro
                                            if (g2,r) in hydroplantto )
                              for b in block )
              + ENDSTORAGE[r] 
                              
            ==
                grb.quicksum( inflow[r] * blkhour[w,b] * 3600
                              for b in block 
                              if r in inflow )
              + startstorage[r] 
                
                for r in reservoir 
            ),  name = 'hydroreservoirservation')



        # Calculate end storage in GWh for last week
        if w == week[-1]:
            m.addConstr(
                (   grb.quicksum( ENDSTORAGE_GWH[vs]
                                  for vs in h2osegment)
                ==
                    grb.quicksum( ENDSTORAGE[r] 
                                  * reservoirfactor[r] / 3.6e6
                                  for r in reservoir)
                ),  name = 'endstoragegwh')

        # Create constraints end ##############################################


        m.optimize()
        #m.write('forward%s.lp' % (w))

        endstorage = m.getAttr('X', ENDSTORAGE)

    """ FORWARD SOLVE END """



    """ BACKWARD SOLVE """
#    """ 
    sce = scenario[0:s]
    for w in weekr:
        
        #  Set model parameters
        inflow = sampledinflow[ (sampledinflow['scenario'] <= s) \
                              & (sampledinflow['week'] == w)]
        inflow = inflow.set_index(['scenario','lake'])
        inflow = dict(inflow['inflow'])
        
        if w < week[-1]:
            slope = slopes[slopes['week']==w].set_index(['reservoir','iter'])
            slope = dict(slope['value'])

            intercept = intercepts[intercepts['week']==w].set_index(['iter'])
            intercept = dict(intercept['value'])

            iteration = intercepts[intercepts['week']==w]['iter'].tolist()
            iteration = [int(i) for i in iteration]

        # Create optimization model
        m = grb.Model('HydroScheduling')


        # Create variables ####################################################
        
        upperbound = [ generator_capacity[g] for sc in sce
                       for b in block for g in generator ]
        GENERATION = m.addVars(sce,block,generator, 
                               ub=upperbound, name='generation')

        upperbound = [ shortageprop[n,vs] * demandmw[w,b,n] for sc in sce
                       for b in block for n in node for vs in vollsegment ]
        SHORTAGE_MW = m.addVars(sce, block, node, vollsegment,
                                ub = upperbound, name='shortage_mw')

        upperbound1 = [ tx_capacity[br] for sc in sce 
                       for b in block for br in tx]
        POWERFLOW = m.addVars(sce, block, tx, 
                              ub = upperbound, name='powerflow')

        TURBINEH2OFLOW = m.addVars(sce, block, generatorhydro, 
                                   name='turbineh2oflow')

        upperbound = [ hydroplant_maxspill[g] if g in hydroplant_maxspill 
                       else grb.GRB.INFINITY for sc in sce
                       for b in block for g in generatorhydro ]
        SPILLEDH2OFLOW = m.addVars(sce, block, generatorhydro,
                                   ub=upperbound, name='spilledh2oflow')
        if w < week[-1]:
            FUTURECOST = m.addVars(sce, name = 'futurecost')
        
        ARCFLOW = m.addVars(sce, block, arc, name='arcflow')

        ARCFLOWDEFICIT = m.addVars(sce, block, maxflowarc, 
                                   name='arcflowdeficit')

        ARCFLOWSURPLUS = m.addVars(sce, block, minflowarc, 
                                   name='arcflowsurplus')

        upperbound = [reservoir_capacity[r] for sc in sce for r in reservoir]
        ENDSTORAGE = m.addVars(sce, reservoir, 
                               ub = upperbound, name='endstorage')

        if w == week[-1]:
            ENDSTORAGE_GWH = m.addVars(sce, h2osegment, 
                                       name ='endstorage_gwh')

        TOTALCOST = m.addVars( sce, lb = -grb.GRB.INFINITY,
                               ub = grb.GRB.INFINITY, name='endstorage')                                       
        # Create variables end ################################################



        # Create objective function ###########################################
        m.setObjective( grb.quicksum( TOTALCOST[sc] for sc in sce) )
        


        # Create constraints ##################################################

        # System cost constraint
        if w < week[-1]:
            m.addConstrs(
                (   TOTALCOST[sc]
                ==
                    grb.quicksum( GENERATION[sc,b,g] 
                                  * srmc[w,g] * blkhour[w,b]
                                  for b in block 
                                  for g in generatorthermal )
                  + grb.quicksum( SHORTAGE_MW[sc,b,n,ls] 
                                  * shortagecost[n,ls] * blkhour[w,b]
                                  for b in block 
                                  for n in node 
                                  for ls in vollsegment )
                  + grb.quicksum( ARCFLOWDEFICIT[sc,b,i,j]
                                  * maxflowcost * blkhour[w,b]
                                  for b in block 
                                  for (i,j) in maxflowarc )
                  + grb.quicksum( ARCFLOWSURPLUS[sc,b,i,j]
                                  * minflowcost * blkhour[w,b]
                                  for b in block 
                                  for (i,j) in minflowarc)
                  + FUTURECOST[sc] 
                    for sc in sce 
                ), name = 'systemcost' )
        else:
            m.addConstrs(
                (   TOTALCOST[sc]
                ==
                    grb.quicksum( GENERATION[sc,b,g]
                                  * srmc[w,g] * blkhour[w,b]
                                  for b in block 
                                  for g in generatorthermal )
                  + grb.quicksum( SHORTAGE_MW[sc,b,n,ls] 
                                  * shortagecost[n,ls] * blkhour[w,b]
                                  for b in block 
                                  for n in node 
                                  for ls in vollsegment )
                  + grb.quicksum( ARCFLOWDEFICIT[sc,b,i,j]
                                  * maxflowcost * blkhour[w,b]
                                  for b in block 
                                  for (i,j) in maxflowarc )
                  + grb.quicksum( ARCFLOWSURPLUS[sc,b,i,j]
                                  * minflowcost * blkhour[w,b]
                                  for b in block 
                                  for (i,j) in minflowarc)
                  - grb.quicksum( ENDSTORAGE_GWH[sc,vs]
                                  * h2odlrpermwh[vs] * 1e3
                                  for vs in h2osegment )
                    for sc in sce
                ),  name = 'systemcost' )

        # Future cost constraints
        if w < week[-1]:
            m.addConstrs(
                (   FUTURECOST[sc] 
                >= 
                    intercept[i] + grb.quicksum( ENDSTORAGE[sc,r]
                                                 * slope[r,i]
                                                 for r in reservoir )
                    for sc in sce
                    for i in iteration
                ),  name = "futurecost")

        # Supply demand balance constrant
        m.addConstrs(
            (   grb.quicksum( GENERATION[sc,b,g]
                              for g,n in generatornode.select('*',n))
              + grb.quicksum( POWERFLOW[sc,b,fr,n]
                              for fr,n in tx.select('*',n))
              - grb.quicksum( POWERFLOW[sc,b,n,to]
                              for n,to in tx.select(n,'*'))
              + grb.quicksum( SHORTAGE_MW[sc,b,n,ls]
                              for ls in vollsegment)
            ==  
                netdemand[w,b,n]
                for sc in sce
                for b in block 
                for n in node
            ),  name = 'supplydemandbalance')

        # Hydro generation conversion
        m.addConstrs(
            (   GENERATION[sc,b,g] 
            == 
                TURBINEH2OFLOW[sc,b,g] * hydroplant_powrfact[g]
                for sc in sce
                for b in block 
                for g in generatorhydro
            ),  name = 'hydrogenerationconversion')

        # Maximum hydro flow constraint
        m.addConstrs(
            (   ARCFLOW[sc,b,i,j] - ARCFLOWDEFICIT[sc,b,i,j]
            <=  
                maxflow[i,j]
                for sc in sce
                for b in block 
                for (i,j) in maxflowarc
            ),  name = 'maxhydroflow')

        # Minimum hydro flow constraint
        m.addConstrs(
            (   ARCFLOW[sc,b,i,j] + ARCFLOWSURPLUS[sc,b,i,j]
            >=  
                minflow[i,j]
                for sc in sce
                for b in block 
                for (i,j) in minflowarc
            ),  name = 'minhydroflow')

        # Hydro junction conservation constraint
        m.addConstrs(
            (   grb.quicksum( ARCFLOW.sum(sc,b,j,'*') * blkhour[w,b] * 3600
                            + grb.quicksum( TURBINEH2OFLOW[sc,b,g1]
                                            * blkhour[w,b] * 3600
                                          + SPILLEDH2OFLOW[sc,b,g1]
                                            * blkhour[w,b] * 3600
                                            for g1 in generatorhydro
                                            if (g1,j) in hydroplantfr )
                              for b in block )
              - grb.quicksum( ARCFLOW.sum(sc,b,'*',j) * blkhour[w,b] * 3600
                            + grb.quicksum( TURBINEH2OFLOW[sc,b,g2] 
                                            * blkhour[w,b] * 3600
                                          + SPILLEDH2OFLOW[sc,b,g2] 
                                            * blkhour[w,b] * 3600
                                            for g2 in generatorhydro
                                            if (g2,j) in hydroplantto )
                              for b in block )
            ==
                grb.quicksum( inflow[sc,j] * blkhour[w,b] * 3600
                              for b in block 
                              if (sc,j) in inflow )
                for sc in sce
                for j in junction 
                if j <> 'SEA' and j not in reservoir
            ),  name = 'hydrojunctionconservation')           
        
        # Hydro reservoir conservation constraint
        RESERVOIR = m.addConstrs(
            (   grb.quicksum( ARCFLOW.sum(sc,b,r,'*') * blkhour[w,b] * 3600
                            + grb.quicksum( TURBINEH2OFLOW[sc,b,g1]
                                            * blkhour[w,b] * 3600
                                          + SPILLEDH2OFLOW[sc,b,g1]
                                            * blkhour[w,b] * 3600
                                            for g1 in generatorhydro
                                            if (g1,r) in hydroplantfr )
                              for b in block )
              - grb.quicksum( ARCFLOW.sum(sc,b,'*',r) * blkhour[w,b] * 3600
                            + grb.quicksum( TURBINEH2OFLOW[sc,b,g2] 
                                            * blkhour[w,b] * 3600
                                          + SPILLEDH2OFLOW[sc,b,g2] 
                                            * blkhour[w,b] * 3600
                                            for g2 in generatorhydro
                                            if (g2,r) in hydroplantto )
                              for b in block )
              + ENDSTORAGE[sc,r] 
            ==
                grb.quicksum( inflow[sc,r] * blkhour[w,b] * 3600
                              for b in block )
              + storedstartstorage[w,r]
                for sc in sce
                for r in reservoir
            ),  name = 'hydroreservoirconservation')

        # Calculate end storage in GWh for last week
        if w == week[-1]:
            m.addConstrs(
                (   grb.quicksum( ENDSTORAGE_GWH[sc,vs]
                                  for vs in h2osegment)
                ==
                    grb.quicksum( ENDSTORAGE[sc,r] 
                                  * reservoirfactor[r] / 3.6e6
                                  for r in reservoir)
                    for sc in sce
                ),  name = 'endstoragegwh')

        # Create constraints end ##############################################

        m.optimize()
        #m.write('backward%s.lp' % (w))
        
        # Calculate slopes and intercepts for expected future cost function
        
        a = m.getAttr('Pi',RESERVOIR)
        a = dict(a)
        a = pd.Series(a,name = 'shadowprice')
        a.index.names = ['scenario','reservoir']
        a = a.reset_index()
        a = a.groupby(['reservoir'])['shadowprice'].mean()
        
        y = m.getAttr('X',TOTALCOST)
        y = dict(y)
        y = pd.Series(y,name = 'totalcost')
        y = y.mean()
        
        ax = 0         
        for r in reservoir:
            slopes = slopes.append({'week':w-1,'reservoir':r,
                                    'iter':s,'value':a[r]},
                                    ignore_index=True)
            ax = ax + a[r] * storedstartstorage[w,r]
            
        b = y - ax
               
        intercepts = intercepts.append({'week':w-1,'iter':s,'value':b},
                                        ignore_index=True)



#    """











