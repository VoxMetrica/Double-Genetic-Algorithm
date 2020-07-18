#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from datetime import datetime
from math import isnan
import warnings
import re
import random
from sklearn.preprocessing import StandardScaler
from itertools import combinations
from statsmodels.tsa.stattools import adfuller
from statsmodels.api import OLS

# ------------------------------------------------------- DEPRECATED FUNCTIONS -------------------------------------------------------- #

# def lonely_lambda(series,cutoff,gl):
    # '''
    # lonely lambda as a function meant to apply to a dataframe, which uses a cutoff and a gl greater or 
    # lesser determiner in order to transform the dataframe into a boolean dataframe whilst preserving the
    # nans (unlike a simple dataframe filter).
    # '''
    # For each column series, I drop the nans and then apply the greater or less than condition,
    # according to the value of gl. This means that, when the series are grouped back together in a
    # dataframe, there will still be nans aside for those rows where every value is a nan. 
    # if gl == 1:
        # return series.dropna() > cutoff
    # elif gl == 0:
        # return series.dropna() < cutoff
    # else:
        # print('gl out of bounds')

# def and_lambda(series,other_dfs,cutoffs,gls):
    # '''
    # and_lambda is supposed to apply to a dataframe and will return a dataframe filter where each value in
    # each dataframe (from the other_dfs iterable and the df to which the function is applied) satisfies
    # being greater than or less than the cutoff, according to the gls iterable.
    # '''
    # Since this function is to be applied to a primary dataframe, it is asymmetrical in nature.
    # Therefore I need to assign the cutoff and greater or lesser determiner for the first dataframe.
    # cutoff1 = cutoffs[0]
    # gl1 = gls[0]
    # I use these values to create an initial filter for the stock under consideration. I am careful to
    # drop nans because otherwise they will masquerade as False boolean values.
    # if gl1 == 1:
        # filtr = (series.dropna() > cutoff1)
    # elif gl1 == 0:
        # filtr = (series.dropna() < cutoff1)
    # else:
        # raise ValueError('gl is not a 1 or a 0')
    # Then I use the other dataframes, cutoffs, and gls to update the filter for that stock
    # for df,cutoff,gl in zip(other_dfs,cutoffs[1:],gls[1:]):
        # if gl == 1:
            # filtr = filtr & (df[series.name].dropna() > cutoff)
        # elif gl == 0:
            # filtr = filtr & (df[series.name].dropna() < cutoff)
        # else:
            # raise ValueError('gl is not a 1 or a 0')
    # I then return this filter. Since this function is applied as a lambda, each stock will get a filter
    # and the result will be a dataframe of appropriate filters.
    # return filtr

# def filter_and(df_list,cutoff_list,gl_list):
    # '''
    # filter_and simplifies and makes more robust the application of and_lambda. It takes in a list of
    # dataframes, cutoffs, and gls. It then checks to see if these lists are all of length one, and if so,
    # uses the lonely_lambda function to the dataframe. If the lists are the same length and greater than
    # one, I make sure all the dataframes have the same columns, and then apply and_lambda
    # '''
    # I check if the lists are all of length one.
    # if len(df_list) == len(cutoff_list) == len(gl_list) == 1:
        # If so, I apply and return the result of lonely_lambda application
        # return df_list[0].apply(lambda x: lonely_lambda(x,cutoff_list[0],gl_list[0]))
    # If the lists are greater than one and equal in lengt...
    # elif len(df_list) == len(cutoff_list) == len(gl_list) > 1:
        # ... I set one of the dfs as the first df to which the and_lambda will be applied.
        # first = df_list[0]
        # I then set the other dfs list and make sure that they have the same columns.
        # other_dfs = df_list[1:]
        # other_dfs = [df[[col for col in first.columns if col in df.columns]] for df in other_dfs]
        # Then I apply and_lambda to the first dataframe.
        # return first.apply(lambda x: and_lambda(x,other_dfs,cutoff_list,gl_list))
    # else:
        # If the arguments are of different length or are equal to 0, I raise an error.
        # raise ValueError('inputted lists do not have the correct or equal lengths')
        
# def useless(df,signal_col='signal',price_col='price',max_limit=5,proportion_limit=0.1):
    # '''
    # This function takes in a dataframe, with one signal column and another price column. The
    # function produces a list of the lengths of consecutive nans. Starting from the first non nan value,
    # the function then determines if either the maximum consecutive number of nans or the proportion of 
    # values that are nans exceeds the given limits, and returns a True boolean if that is so.
    # '''
    # First I check to see if there is only one column in the dataframe. If so, this means I either
    # don't have signal or price data, and therefore the stonk is useless, so I return True.
    # if len(df.columns) == 1:
        # return True
    # If dropna() gives an empty dataframe, then obviously it is useless, so I return True.
    # if len(df.dropna()) == 0:
        # return True
    # These checks out of the way, I find the first non nan index value
    # start = df.dropna().index[0]
    # I create the isNaN boolean series, which identifies when there is an nan in either the prices or
    # signals dataframe.
    # isNaN = df.loc[start:].isna()
    # isNaN = isNaN[signal_col] | isNaN[price_col]
    # I turn the isNaN series into a string. of ones and zeros
    # string = str(isNaN.apply(lambda x: 1 if x else 0).tolist()).replace('[','').replace(
        # ']','').replace(',','').replace(' ','')
    # Then I use re's findall function to return a list of all the consecutive 1s and take the len of the
    # consecutive 1s strings so that I am left with a list of numbers representing consecutive nan
    # lengths.
    # consecs = [len(match) for match in re.findall('1+',string)]
    # I then check if the maximum consecutive nan length or nan proportion is greater than given cuttoffs.
    # return robust_max(consecs) >= max_limit or (sum(isNaN) / len(isNaN)) >= proportion_limit
    
# def signal_from_guide(buy_technicals,buy_cutoffs,buy_gls,sell_technicals,sell_cutoffs,sell_gls):
    # '''
    # This function takes the technicals dataframes, cutoffs, and gls, and uses them to produce a signals
    # dataframe that has a 1 with a buy signal and -1 with a sell signal. 0 is a null signal. It does so
    # whilst removing those stocks where signals are too sparse. Look at the 'droppers' function for more
    # details.
    # '''
    # I use the signal_maker function to get buy and sell binary dataframes indicating the when buy or
    # sell signals are active. I subtract the sell signal from the buy signal to obtain an overall
    # signals dataframe where 1 is a buy signall and -1 is a sell signal.
    # buy_signal = signal_maker(buy_technicals,buy_cutoffs,buy_gls)
    # sell_signal = signal_maker(sell_technicals,sell_cutoffs,sell_gls)
    # signals = buy_signal - sell_signal
    # I pair the signals array with the actionable prices and then use the useless function to
    # identify stonks with large actionable gaps, and drop these columns.
    # exclude = drop_false(signals.apply(droppers)).index
    # signals = signals[[col for col in signals.columns if col not in exclude]]
    # Then I concatenate the remaining signals whilst dropping any nans column by column, so that I am
    # left with only rows where there is at least one signal. I don't use dropna() on the whole dataframe
    # because that would throw away useful data for other stocks as long as there were one nan.
    # return signals[signals.first_valid_index():]

# def slow_signal_from_guide(buy_technicals,buy_cutoffs,buy_gls,sell_technicals,sell_cutoffs,sell_gls,
                           # action_prices):
    # '''
    # This function is like that above, except that it removes stocks where both signal and price paired 
    # together are too sparse, rather than looking only at teh sparsity of signals. Look at the
    # 'useless' function for more details.
    # '''
    # I use the signal_maker function to get buy and sell binary dataframes indicating the when buy or
    # sell signals are active. I subtract the sell signal from the buy signal to obtain an overall
    # signals dataframe where 1 is a buy signall and -1 is a sell signal.    
    # buy_signal = signal_maker(buy_technicals,buy_cutoffs,buy_gls)
    # sell_signal = signal_maker(sell_technicals,sell_cutoffs,sell_gls)
    # signals = buy_signal - sell_signal
    # I pair the signals array with the actionable prices and then use the useless function to
    # identify stonks with large actionable gaps, and drop these columns.
    # together = pd.concat([signals,action_prices],keys=['signal','price'],axis=1)
    # bad_cols = [col for col in together.columns.levels[1] if useless(together.xs(col,axis=1,level=1))]
    # together.drop(bad_cols,axis=1,level=1,inplace=True)
    # together.columns = together.columns.remove_unused_levels()
    # Then I concatenate the remaining signals whilst dropping any nans column by column, so that I am
    # left with only rows where there is at least one signal. I don't use dropna() on the whole dataframe
    # because that would throw away useful data for other stocks as long as there were one nan.
    # return pd.concat([together.xs(col,axis=1,level=1).dropna()['signal'] for col in 
                         # together.columns.levels[1]],axis=1,keys=together.columns.levels[1])
                         
# def rigorous_fitness(chromosome,prices,closes,action_prices,ma_keys,bol_keys,
                     # growth_keys,supmax_keys,supmin_keys,return_keys,pvma_keys,delta_context,
                     # price_context,pv_context,min_density=0.5,spread=0.09/100,min_invest=50,
                     # start=5000):
    # '''
    # Just like fitness except that it uses slow_signal_from_guide instead of signal_from_guide.
    # '''
    # I split the chromosome into its constituent components.
    # buy_chromosome, sell_chromosome, num_stocks = chromosome
    # I use the get_measures function to get the lists of technical dataframes, cutoffs, and gls.
    # buy_technicals,buy_cutoffs,buy_gls,sell_technicals,sell_cutoffs,sell_gls = get_measures(
        # buy_chromosome,sell_chromosome,prices,closes,ma_keys,bol_keys,growth_keys,supmax_keys,
        # supmin_keys,return_keys,pvma_keys,delta_context,price_context,pv_context)
    # Supressing some useless warnings, I standardise the technicals dataframes.
    # with warnings.catch_warnings():
        # warnings.simplefilter("ignore")
        # buy_technicals = [pd.DataFrame(StandardScaler().fit_transform(tech),index=tech.index,
                                       # columns=tech.columns) 
                                       # for tech in buy_technicals] 
        # sell_technicals = [pd.DataFrame(StandardScaler().fit_transform(tech),index=tech.index,
                                        # columns=tech.columns) 
                                        # for tech in sell_technicals]
    # Using the resulting technicals, I use slow_signal_from_guide to produce a signals dataframe.
    # signals = slow_signal_from_guide(buy_technicals,buy_cutoffs,buy_gls,
                                     # sell_technicals,sell_cutoffs,sell_gls,
                                    # action_prices)
    
    # If there are no buy signals, or there are too few signals relative to prices, I return 0.
    # if len(signals) / len(action_prices) < min_density or (signals==1).sum().sum() == 0:
        # return 0
    # Then I create the actions dictionary with action_maker...
    # actions = action_maker(signals)
    # ...and run the simulation, returning its result.
    # return simulation(actions, action_prices, start, num_stocks, min_invest, spread, signals.index[0])
    
# def old_simulation(actions,action_prices,start,num_stocks,min_invest,spread,first_signal):
    # '''
    # simulation takes an actions dictionary and other arguments to run a simulaltion where I buy and sell
    # stocks. The simulation returns the daily compound return calculated by taking the total return
    # and rooting to the order of the time delta between the first signal and the last price. All the
    # arguments are explained in the fitness function, except for actions (explained just above) and
    # first_signal, which is the timestamp representing the first valid signal.
    # '''
    # available = start
    # portfolio = {}
    # I start a for loop with the dates in order.
    # for date in actions:
        # buy is a list of stocks that the signals tell me to buy, and active_sell is the stocks for
        # which there are sell signals and that are already in the portfolio.
        # buy = actions[date]['buy']
        # sell = actions[date]['sell']
        # active_sell = [stonk for stonk in sell if stonk in portfolio.keys()]
        # If active_sell has some stonks, then sell action takes place...
        # if len(active_sell) > 0:
            # ...For each stonk in active sells, I find the selling price and divide it by the purchase
            # price of the stonk as specified in the portfolio. I multiply this return by the amount
            # invested in that stonk. I sum over the sold stonks and add it to the available amount.
            # available = available + sum([(action_prices[stonk].loc[date] / portfolio[stonk]['price']) * 
                                         # portfolio[stonk]['amount'] for stonk in active_sell])
            # Then I delete the stonks in the portfolio.
            # for key in active_sell:
                # del portfolio[key]
        # else:
            # pass
        # Then I define the 'space' variable, which indicates the number of stocks that can be invested
        # in at that moment. It is the minimum of [the maximum number of stocks to invest in minus the
        # current stocks in the portfolio] and [the integer value of the available amount divided by the
        # minimum amount I can invest]. The second argument of this minimum is so that I never invest
        # more than the minimum amount into a stock.
        # space = min(num_stocks - len(portfolio.keys()), int(available / min_invest))        
        # Then, if there are any buy signals, and there is space I proceed - note that if I had less than
        # the minimum amount you can invest, then 'space' would be 0 and I would not proceed.
        # if len(buy) > 0 and space > 0:
            # I define the amount I will put into each stock as the available amount divided by the space
            # variable. This ensures that when I have the maximum number of stocks in my portfolio, I
            # will have spent all my money. It also ensures I never invest more than the minimum amount
            # because of how space is defined.
            # per_stonk = available / space
            # If there are more stocks to buy than space available, I randomly choose stocks to buy.
            # buy = random.sample(buy,k=min(space,len(buy)))
            # I then update the portfolio with the stocks I bought, their purchase price including
            # spread, and the amount invested in each.
            # portfolio.update({stonk:{'amount':per_stonk,'price':(1+spread) * action_prices[stonk].loc[
                # date]} for stonk in buy})
            # I then subtract from available the total cost of the stock purchase.
            # available = available - (per_stonk * len(buy))
        # else:
            # pass
    # After the whole for loop, sum the available with the value of the portfolio as given by the last
    # available price for each stock. These last available prices cannot be for dates that are too far
    # away from each other due to my dropping stocks with sparse data earlier.
    # total = available + sum([(action_prices[stonk].loc[action_prices[stonk].last_valid_index()] / portfolio[stonk]['price']) * portfolio[
                # stonk]['amount'] for stonk in portfolio])
    # Finally, I return the total return as a daily return.
    # return (total / start) ** (1 / (action_prices.last_valid_index() - first_signal).days)

# def old_fitness(chromosome,prices,closes,action_prices,ma_keys,bol_keys,growth_keys,
            # supmax_keys,supmin_keys,return_keys,pvma_keys,delta_context,price_context,pv_context,
            # min_density=0.5,spread=0.09/100,min_invest=50,start=5000):
    # '''
    # This is the fitness function that returns the daily compound return in 1 + x form, owing to the
    # strategy given by the buy and sell chromosomes, and the maximum number of stocks to invest in in any
    # one time. The arguments are:
    # - buy_chromosome: This is a dataframe containing a params column wich details the parameters that
    # make the measures that form the signal, a cutoffs column that defines the cutoff in terms of Zscore
    # for each measure, and a gl column detailing whether the signal is positive when the measure is greater
    # than or less than the given cutoff. The dataframe's index is the name of the measures being employed.
    # - sell_chromosome: just like the buy_chromosome but for the sell signal.
    # - prices: This is the dataframe with high,low,open,close,and volume values.
    # - closes: This is a dataframe of close prices. I have it as an argument to save having to define it 
    # from prices within the function each time.
    # - action_prices: This is the dataframe of prices that will actually be used for buying and selling.
    # Currently, At first, I had it as the open prices. It may still be that, depending on when I am 
    # reading this.
    # - _keys arguments: These arguments contain the names of the measures created with different 
    # calibrations of the base industry Zscore dataframes represented in the following arguments.
    # - _context arguments: These arguments contain the base mega-dataframes with different calibrations
    # of industry Zscore dataframes.
    # - num_stocks: This defines the maximum number of stocks that will be invested in in any one time.
    # - min_density: This defines the minimum proportion of dates for which there must be some signal for
    # some stock, as a proportion of the total number of dates for which there is price data.
    # - spread: This is the proportion of a stock's price that will be added to its buying price.
    # - min_invest: This is the minimum amount that you can put in a stock.
    # - start: This is the initial amount you have to invest.
    # '''
    # I split the chromosome into its constituent components.
    # buy_chromosome, sell_chromosome, num_stocks = chromosome
    # I use the get_measures function to get the lists of technical dataframes, cutoffs, and gls.
    # buy_technicals,buy_cutoffs,buy_gls,sell_technicals,sell_cutoffs,sell_gls = get_measures(
        # buy_chromosome,sell_chromosome,prices,closes,ma_keys,bol_keys,growth_keys,supmax_keys,
        # supmin_keys,return_keys,pvma_keys,delta_context,price_context,pv_context)
    # Supressing some useless warnings, I standardise the technicals dataframes.
    # with warnings.catch_warnings():
        # warnings.simplefilter("ignore")
        # buy_technicals = [pd.DataFrame(StandardScaler().fit_transform(tech),index=tech.index,
                                       # columns=tech.columns) 
                                       # for tech in buy_technicals] 
        # sell_technicals = [pd.DataFrame(StandardScaler().fit_transform(tech),index=tech.index,
                                        # columns=tech.columns) 
                                        # for tech in sell_technicals]
    # Using the resulting technicals, I use signal_from_guide to produce a signals dataframe.
    # signals = signal_from_guide(buy_technicals,buy_cutoffs,buy_gls,sell_technicals,sell_cutoffs,sell_gls)
    
    # If there are no buy signals, or there are too few signals relative to prices, I return 0.
    # if len(signals) / len(action_prices) < min_density or (signals==1).sum().sum() == 0:
        # return 0
    # Then I create the actions dictionary with action_maker...
    # actions = action_maker(signals)
    # ...and run the simulation, returning its result.
    # return old_simulation(actions, action_prices, start, num_stocks, min_invest, spread, signals.index[0])

# def 20200604_fitness(chromosome,prices,closes,action_prices,ma_keys,bol_keys,growth_keys,
            # supmax_keys,supmin_keys,return_keys,pvma_keys,delta_context,price_context,pv_context,
            # min_density=0.5,spread=0.09/100,min_invest=50,start=5000,max_limit=5,proportion_limit=0.1):
    # '''
    # This is the fitness function that returns the daily compound return in 1 + x form, owing to the
    # strategy given by the buy and sell chromosomes, and the maximum number of stocks to invest in in any
    # one time. The arguments are:
    # - buy_chromosome: This is a dataframe containing a params column wich details the parameters that
    # make the measures that form the signal, a cutoffs column that defines the cutoff in terms of Zscore
    # for each measure, and a gl column detailing whether the signal is positive when the measure is greater
    # than or less than the given cutoff. The dataframe's index is the name of the measures being employed.
    # - sell_chromosome: just like the buy_chromosome but for the sell signal.
    # - prices: This is the dataframe with high,low,open,close,and volume values.
    # - closes: This is a dataframe of close prices. I have it as an argument to save having to define it 
    # from prices within the function each time.
    # - action_prices: This is the dataframe of prices that will actually be used for buying and selling.
    # Currently, At first, I had it as the open prices. It may still be that, depending on when I am 
    # reading this.
    # - _keys arguments: These arguments contain the names of the measures created with different 
    # calibrations of the base industry Zscore dataframes represented in the following arguments.
    # - _context arguments: These arguments contain the base mega-dataframes with different calibrations
    # of industry Zscore dataframes.
    # - num_stocks: This defines the maximum number of stocks that will be invested in in any one time.
    # - min_density: This defines the minimum proportion of dates for which there must be some signal for
    # some stock, as a proportion of the total number of dates for which there is price data.
    # - spread: This is the proportion of a stock's price that will be added to its buying price.
    # - min_invest: This is the minimum amount that you can put in a stock.
    # - start: This is the initial amount you have to invest.
    # '''
    
    # I split the chromosome into its constituent components.
    # buy_chromosome, sell_chromosome, num_stocks = chromosome
    # I use the get_measures function to get the lists of technical dataframes, cutoffs, and gls.
    # buy_technicals,buy_cutoffs,buy_gls,sell_technicals,sell_cutoffs,sell_gls = get_measures(
        # buy_chromosome,sell_chromosome,prices,closes,ma_keys,bol_keys,growth_keys,supmax_keys,
        # supmin_keys,return_keys,pvma_keys,delta_context,price_context,pv_context)
    # Supressing some useless warnings, I standardise the technicals dataframes.
    # with warnings.catch_warnings():
        # warnings.simplefilter("ignore")
        # buy_technicals = [pd.DataFrame(StandardScaler().fit_transform(tech),index=tech.index,
                                       # columns=tech.columns) 
                                       # for tech in buy_technicals] 
        # sell_technicals = [pd.DataFrame(StandardScaler().fit_transform(tech),index=tech.index,
                                        # columns=tech.columns) 
                                        # for tech in sell_technicals]
    # Using the resulting technicals, I use OG_signal to produce a signals dataframe.
    # signals = OG_signal(buy_technicals,buy_cutoffs,buy_gls,sell_technicals,sell_cutoffs,sell_gls)

    # If there are no buy signals, or there are too few signals relative to prices, I return np.nan.
    # if len(signals[signals.first_valid_index():]) / len(action_prices
                                                       # ) < min_density or (signals==1).sum().sum() == 0:
        # return np.nan
    # I then refine the signals and action_prices dataframes to include only relevant information.
    # signals,action_prices = refiner(signals, action_prices, max_limit, proportion_limit)
    # Then I create the actions dictionary with action_maker...
    # actions = action_maker(signals)
    # ...and run the simulation, returning its result.
    # return simulation(actions, action_prices, start, num_stocks, min_invest, spread, signals.index[0])
    
# def saved_fitness(chromosome,prices,closes,action_prices,ma_keys,bol_keys,growth_keys,
            # supmax_keys,supmin_keys,return_keys,pvma_keys,delta_context,price_context,pv_context,
            # min_density=0.5,spread=0.09/100,min_invest=50,start=5000,max_limit=5,proportion_limit=0.1):
    # '''
    # This is the fitness function that returns the daily compound return in 1 + x form, owing to the
    # strategy given by the buy and sell chromosomes, and the maximum number of stocks to invest in in any
    # one time. The arguments are:
    # - buy_chromosome: This is a dataframe containing a params column wich details the parameters that
    # make the measures that form the signal, a cutoffs column that defines the cutoff in terms of Zscore
    # for each measure, and a gl column detailing whether the signal is positive when the measure is greater
    # than or less than the given cutoff. The dataframe's index is the name of the measures being employed.
    # - sell_chromosome: just like the buy_chromosome but for the sell signal.
    # - prices: This is the dataframe with high,low,open,close,and volume values.
    # - closes: This is a dataframe of close prices. I have it as an argument to save having to define it 
    # from prices within the function each time.
    # - action_prices: This is the dataframe of prices that will actually be used for buying and selling.
    # Currently, At first, I had it as the open prices. It may still be that, depending on when I am 
    # reading this.
    # - _keys arguments: These arguments contain the names of the measures created with different 
    # calibrations of the base industry Zscore dataframes represented in the following arguments.
    # - _context arguments: These arguments contain the base mega-dataframes with different calibrations
    # of industry Zscore dataframes.
    # - num_stocks: This defines the maximum number of stocks that will be invested in in any one time.
    # - min_density: This defines the minimum proportion of dates for which there must be some signal for
    # some stock, as a proportion of the total number of dates for which there is price data.
    # - spread: This is the proportion of a stock's price that will be added to its buying price.
    # - min_invest: This is the minimum amount that you can put in a stock.
    # - start: This is the initial amount you have to invest.
    # '''
    
    # I split the chromosome into its constituent components.
    # buy_chromosome, sell_chromosome, num_stocks = chromosome
    # I use the get_measures function to get the lists of technical dataframes, cutoffs, and gls.
    # buy_technicals,buy_cutoffs,buy_gls,sell_technicals,sell_cutoffs,sell_gls = get_measures(
        # buy_chromosome,sell_chromosome,prices,closes,ma_keys,bol_keys,growth_keys,supmax_keys,
        # supmin_keys,return_keys,pvma_keys,delta_context,price_context,pv_context)
    # Supressing some useless warnings, I standardise the technicals dataframes.
    # with warnings.catch_warnings():
        # warnings.simplefilter("ignore")
        # buy_technicals = [pd.DataFrame(StandardScaler().fit_transform(tech),index=tech.index,
                                       # columns=tech.columns) 
                                       # for tech in buy_technicals] 
        # sell_technicals = [pd.DataFrame(StandardScaler().fit_transform(tech),index=tech.index,
                                        # columns=tech.columns) 
                                        # for tech in sell_technicals]
    # Using the resulting technicals, I use OG_signal to produce a signals dataframe. I also
    # set the first valid index of the signals dataframes as first_signal.
    # signals = OG_signal(buy_technicals,buy_cutoffs,buy_gls,sell_technicals,sell_cutoffs,sell_gls)
    # first_signal = signals.first_valid_index()

    # If there are no buy signals, or there are too few signals relative to prices, or there are 
    # fewer than zero days between the first valid signal and the last available price date, then 
    # I return np.nan.
    # if (len(signals[first_signal:]) / len(action_prices) < min_density or (signals==1).sum().sum() == 0 
        # or (action_prices.last_valid_index() - first_signal).days <= 0):
        # return np.nan
    # I then refine the signals and action_prices dataframes to include only relevant information.
    # signals,action_prices = refiner(signals, action_prices, max_limit, proportion_limit)
    # Then I create the actions dictionary with action_maker...
    # actions = action_maker(signals)
    # ...and run the simulation, returning its result.
    # return simulation(actions, action_prices, start, num_stocks, min_invest, spread, first_signal)
    
# def diagnostic_validation_fitness(complete_solution,prices,closes,action_prices,ma_keys,bol_keys,growth_keys,
            # supmax_keys,supmin_keys,return_keys,pvma_keys,delta_context,price_context,pv_context,
            # min_density=0.5,spread=0.09/100,min_invest=50,start=5000,max_limit=5,proportion_limit=0.1):
    # '''
    # This fitness function takes a complete solution, which is a genome plus standard scaler objects and
    # column indexes that contextualise the technicals dataframes, and returns the fitness for that complete
    # solution. This fitness function must be set to have test data in order for it to return a useful
    # result.
    # '''
    
    # I split the complete_solution into two components: The genetic component, and the scalers and 
    # columns component. These are subsequently split into their constituent components.
    # buy_chromosome, sell_chromosome, num_stocks = complete_solution[0]
    # buy_scalers,buy_columns,sell_scalers,sell_columns = complete_solution[1]
    
    # I use the get_measures function to get the lists of technical dataframes, cutoffs, and gls.
    # buy_technicals,buy_cutoffs,buy_gls,sell_technicals,sell_cutoffs,sell_gls = get_measures(
        # buy_chromosome,sell_chromosome,prices,closes,ma_keys,bol_keys,growth_keys,supmax_keys,
        # supmin_keys,return_keys,pvma_keys,delta_context,price_context,pv_context)
    
    # Supressing some useless warnings, I standardise the technicals dataframes using the standard
    # scalers contained in the complete solution. The columns are used to ensure that the standard
    # scaler objects are applied to the correct columns.
    # with warnings.catch_warnings():
        # warnings.simplefilter("ignore")
        # new_buy_technicals, new_sell_technicals = [],[]
        # for i in range(len(buy_scalers)):
            # try:
                # appender = pd.DataFrame(buy_scalers[i].transform(
                    # buy_technicals[i][buy_columns[i]]),
                    # index=buy_technicals[i].index,
                    # columns=buy_columns[i])
                # new_buy_technicals.append(appender)
            # except KeyError:
                # print(f'*buy* loop iteration: {i}','\n')
                # print(buy_chromosome['params'],'\n')
                # print(f'column index type: {type(buy_technicals[i].columns)}','\n')
                # print(f'missing columns: {[col for col in buy_columns[i] if col not in buy_technicals[i].columns]}')
                # print('-----------------------------------------------------------------------------')
                # new_tech = pd.DataFrame(buy_technicals[i],columns=buy_columns[i].columns)
                # appender = pd.DataFrame(buy_scalers[i].transform(
                    # new_tech[buy_columns[i]]),
                    # index=buy_technicals[i].index,
                    # columns=buy_columns[i])
                # new_buy_technicals.append(appender)
            # try:
                # appender = pd.DataFrame(sell_scalers[i].transform(
                    # sell_technicals[i][sell_columns[i]]),
                    # index=sell_technicals[i].index,
                    # columns=sell_columns[i])
                # new_sell_technicals.append(appender)
            # except KeyError:
                # print(f'*sell* loop iteration: {i}','\n')
                # print(sell_chromosome['params'],'\n')
                # print(f'column index type: {type(sell_technicals[i].columns)}','\n')
                # print(f'missing columns: {[col for col in sell_columns[i] if col not in sell_technicals[i].columns]}')
                # print('-----------------------------------------------------------------------------')
                # new_tech = pd.DataFrame(sell_technicals[i],columns=sell_columns[i].columns)
                # appender = pd.DataFrame(sell_scalers[i].transform(
                    # new_tech[sell_columns[i]]),
                    # index=sell_technicals[i].index,
                    # columns=sell_columns[i])
                # new_sell_technicals.append(appender)
        
        # buy_technicals, sell_technicals = new_buy_technicals, new_sell_technicals
            
    
    # Using the resulting scaled technicals, I use OG_signal to produce a signals dataframe. I also
    # set the first valid index of the signals dataframes as first_signal.
    # signals = OG_signal(buy_technicals,buy_cutoffs,buy_gls,sell_technicals,sell_cutoffs,sell_gls)
    # first_signal = signals.first_valid_index()

    # If there are no buy signals, or there are too few signals relative to prices, or there are 
    # fewer than zero days between the first valid signal and the last available price date, then 
    # I return np.nan.
    # if (len(signals[first_signal:]) / len(action_prices) < min_density or (signals==1).sum().sum() == 0 
        # or (action_prices.last_valid_index() - first_signal).days <= 0):
        # return np.nan
    # I then refine the signals and action_prices dataframes to include only relevant information.
    # signals,action_prices = refiner(signals, action_prices, max_limit, proportion_limit)
    # Then I create the actions dictionary with action_maker...
    # actions = action_maker(signals)
    # ...and run the simulation, returning its result.
    # return simulation(actions, action_prices, start, num_stocks, min_invest, spread, first_signal)

# -------------------------------------------------------- ACTIVE FUNCTIONS ----------------------------------------------------------- #

def drop_false(series):
    '''
    This function takes a series and drops any indices with False boolean values
    '''
    return series.apply(lambda x: x if x else np.nan).dropna()
    
def drop_true(series):
    '''
    This function takes a series and drops any indices with False boolean values
    '''
    return series.apply(lambda x: np.nan if x else x).dropna()

def binarise(bool_num):
    '''
    This function takes boolean values and transforms Trues into ones and Falses into zeros. Any nans are
    keapt as nans.
    '''
    if isnan(bool_num):
        return np.nan
    elif bool_num:
        return 1
    else:
        return 0

def interise(num):
    '''
    This function turns any floats into integers, whilst keeping nans as nans.
    '''
    if isnan(num):
        return np.nan
    elif type(num) == float:
        return int(num)
    elif type(num) == int:
        return num
    else:
        raise ValueError('Values in series are not floats or ints.')

def stringarise(series):
    '''
    This function takes a series of ones, zeros, minus ones, and nans, and turns it into a tightly packed
    string. the string contains only '1', '0', 's', 'n', and 'e.' the ones signify a buy signal. 's'
    signifies a sell signal. zeros are ambiguous or null signalls. 'n' is for nans, and 'e,' which always
    goes at the end, represents the end of the string. The whole string codes the series sequentially
    and regex can be used to identify the iloc of buy and sell actions within the context of the series'
    index.
    '''
    return str([interise(n) for n in series]).replace('nan','n').replace('-1','s').replace(
        '[','').replace(']','').replace(',','').replace(' ','') + 'e'

def robust_max(iterable):
    '''
    robust_max is just a normal max function but it returns 0 when given an empty list.
    '''
    if len(iterable) == 0:
        return 0
    else:
        return max(iterable)

def droppers(series,max_limit=5,proportion_limit=0.1):
    '''
    droppers identifies, from the first valid index, sparse series. It returns True if the maximum
    consecutive number of nans exceeds the max_limit, or if the proportion of nans is greater than the
    proportion_limit.
    '''
    # I check to see if there is no valid data, and in that case I return a True.
    if len(series.dropna()) == 0:
        return True
    else:
        # I find the first valid index and then stringarise the series after that index into a string
        # of ones and zeros.
        start = series.first_valid_index()
        isnull = series[start:].isna()
        string = str(isnull.apply(lambda x: 1 if x else 0).tolist()).replace('[','').replace(
        ']','').replace(',','').replace(' ','')
        # Then I use re's findall function to return a list of all the consecutive 1s and take the len of 
        # the consecutive 1s strings so that I am left with a list of numbers representing consecutive nan
        # lengths.
        consecs = [len(match) for match in re.findall('1+',string)]
        # I then check if the maximum consecutive nan length or nan proportion is greater than given 
        # cuttoffs.
        return robust_max(consecs) >= max_limit or (sum(isnull) / len(isnull)) >= proportion_limit

def buy_dates(string,index):
    '''
    buy_dates takes a string as produced by the stringarise function and uses the re to find the ilocs
    within the string that represent buy signals. This is done by matching substrings that start with the
    first 1 or the first 1 following an s, and ends with the first s following the 1. The function also
    matches the substring following the first 1 after an s (or the first 1 simpliciter) and ends in the e 
    that always ends the string. The iloc of the start of these substrings within the whole string is
    used to retrieve the date in which the actionable buy signal is given. A list of these actionable
    buy dates is returned.
    '''
    p = re.compile('1[0-1n]*s')
    e = re.compile('1[0-1n]*e')
    return [index[m.start()] for m in p.finditer(string)] + [index[m.start()] for m in e.finditer(
        string)]

def sell_dates(string,index):
    '''
    sell_dates takes a string as produced by the stringarise function and uses the re to find the ilocs
    within the string that represent buy signals. This is done by matching substrings that start with the
    first 1 or the first 1 following an s, and ends with the first s following the 1. the iloc of the end
    is used to retrieve the date of the sell signal from the index. A list of these dates is returned.
    '''
    p = re.compile('1[0-1n]*s')
    return [index[m.end() - 1] for m in p.finditer(string)]
    
def level_finder(levels,columns):
    '''
    This finds the level of a multi-index's levels in which the given columns reside. If the columns are
    not there in their entirety, it returns None.
    '''
    for i in range(len(levels)):
        if set(columns).issubset(set(levels[i])):
            return i

def borg_split(proportion=0.7,output='tr',max_limit=5,proportion_limit=0.1,*dfs):
    '''
    This function splits the dataframes inputted according to the proportion inputted. The proportion 
    argument determines which proportion of the length of the index of the *first* inputted dataframe
    is used to determine the date index at which data is split into training and test data. The first
    inputted dataframe is also used to determine useless stock columns, and removes them from each 
    dataframe, where either the maximum number of consecutive nans is greater than max_limit, 
    or the overall proportion is greater than proportion limit, counting from the first valid index. 
    The output argument can be set to 'b' for both training and test outputs, 'tr' for just training 
    data, and 'te' for just test data. The function returns a list of the transformed dataframes, with
    the training set first and test set second if 'b' is passed as the output argument. Note that the
    function assumes that if the index is multiindex, then tickers are at level 1.
    '''
    # I create two empty lists that will be filled with the transformed dataframes
    training_dfs = []
    test_dfs = []
    # I set the first dataframe as 'first' and find the index that represents the proportion split 
    # inputted.
    first = dfs[0]
    training_loc = first.index[int(len(first.index)*proportion)]
    
    # I then use the droppers function to determine which stocks to keep and which to drop. This is
    # calculated on the basis of the nans in the first inputted dataframe
    keeps = drop_true(first.apply(lambda x: droppers(x,max_limit,proportion_limit))).index
    drops = [stock for stock in first.columns if stock not in keeps]
    
    # If output is either 'b' or 'tr,' I transform the dataframes into training data and append them to
    # the training list.  
    if output == 'b' or output == 'tr':
        for df in dfs:
            # ...If the index is not multi, I just refine the dataframe 
            if type(df.columns) == pd.core.indexes.base.Index:
                training_dfs.append(df.loc[:training_loc,keeps])
            # but if it is multi, I refine the dataframe by dropping the drops at the appropriate level.
            else:
                level = level_finder(df.columns.levels,drops)
                appender = df.drop(drops,level=level,axis=1).loc[:training_loc]
                appender.columns = appender.columns.remove_unused_levels()
                training_dfs.append(appender)
    # If output is either 'b' or 'te' I do the same for the test split.
    if output == 'b' or output == 'te':
        for df in dfs:
            if type(df.columns) == pd.core.indexes.base.Index:
                training_dfs.append(df.loc[training_loc:,keeps])
            else:
                level = level_finder(df.columns.levels,drops)
                appender = df.drop(drops,level=level,axis=1).loc[training_loc:]
                appender.columns = appender.columns.remove_unused_levels()
                training_dfs.append(appender)
    # I return the sum of the two lists, so that I can have either one or the other or both.
    return training_dfs + test_dfs

def split_harmonise(proportion=0.7,output='tr',max_limit=5,proportion_limit=0.1,*dfs):
    '''
    This function splits the dataframes inputted according to the proportion inputted. The proportion 
    argument determines which proportion of the length of the index of the *first* inputted dataframe
    is used to determine the date index at which data is split into training and test data. The first
    inputted dataframe is also used to determine useless stock columns, and removes them from each 
    dataframe, where either the maximum number of consecutive nans is greater than max_limit, 
    or the overall proportion is greater than proportion limit, counting from the first valid index. 
    The output argument can be set to 'b' for both training and test outputs, 'tr' for just training 
    data, and 'te' for just test data. The function returns a list of the transformed dataframes, with
    the training set first and test set second if 'b' is passed as the output argument. Note that the
    function assumes that if the index is multiindex, then tickers are at level 1.
    '''
    # I create two empty lists that will be filled with the transformed dataframes
    training_dfs = []
    test_dfs = []
    # I set the first dataframe as 'first' and find the index that represents the proportion split inputted.
    first = dfs[0]
    training_loc = first.index[int(len(first.index)*proportion)]
    # If output is either 'b' or 'tr,' I transform the dataframes into training data and append them to
    # the training list.
    if output == 'b' or output == 'tr':
        # I define the stocks to keep and the complimentary set to drop
        keeps = drop_true(first.loc[:training_loc].apply(
            lambda x: droppers(x,max_limit,proportion_limit))).index
        drops = [stock for stock in first.columns if stock not in keeps]
        # Then for each dataframe...
        for df in dfs:
            # ...If the index is not multi, I just refine the dataframe 
            if type(df.columns) == pd.core.indexes.base.Index:
                training_dfs.append(df.loc[:training_loc,keeps])
            # but if it is multi, I refine the dataframe by dropping the drops at the appropriate level.
            else:
                level = level_finder(df.columns.levels,drops)
                appender = df.drop(drops,level=level,axis=1).loc[:training_loc]
                appender.columns = appender.columns.remove_unused_levels()
                training_dfs.append(appender)
    # If output is either 'b' or 'te' I do the same for the test split.
    if output == 'b' or output == 'te':
        keeps = drop_true(first.loc[training_loc:].apply(
            lambda x: droppers(x,max_limit,proportion_limit))).index
        drops = [stock for stock in first.columns if stock not in keeps]
        for df in dfs:
            if type(df.columns) == pd.core.indexes.base.Index:
                training_dfs.append(df.loc[training_loc:,keeps])
            else:
                level = level_finder(df.columns.levels,drops)
                appender = df.drop(drops,level=level,axis=1).loc[training_loc:]
                appender.columns = appender.columns.remove_unused_levels()
                training_dfs.append(appender)
    # I return the sum of the two lists, so that I can have either one or the other or both.
    return training_dfs + test_dfs

def signal_maker(technicals,cutoffs,gls):
    '''
    signal_maker returns a dataframe of ones, zeros, and nans, which represent a signal to buy or sell.
    To clarify, ones represent either a signal to buy or a signal to sell, depending on whether the
    arguments inputted were from the buy or sell chromosome. The function takes the following arguments:
    - technicals: This is an iterable of technical measure dataframes.
    - cutoffs: This is an iterable of values that represents the cutoff values for each technicals
    dataframe, in the same order as the technicals iterable.
    - gls: This is an iterable of ones and zeros which represents if the filter is a greater than or
    less than filter. It is again in the same order as the other iterables.
    
    '''
    # I set the first item of each iterable to a named variable
    first_tech = technicals[0]
    first_cutoff = cutoffs[0]
    first_gl = gls[0]
    # If the first_gl is one, then I use >= to create the signal. Otherwise I use <=
    if first_gl == 1:
        sig = first_tech >= first_cutoff
    elif first_gl == 0:
        sig = first_tech <= first_cutoff
    else:
        raise ValueError('gl is not 1 or 0')
    # I make sure to preserve the nans of the dataframe. Note that by some dark magic, this process
    # also turns Trues into 1s and Falses into zeros.
    sig[first_tech.isnull()] = np.nan
    # I set up a for loop with the rest of the items in the iterables zipped together.
    for tech,cutoff,gl in zip(technicals[1:],cutoffs[1:],gls[1:]):
        if gl == 1:
            techsig = tech >= cutoff
        elif gl == 0:
            techsig = tech <= cutoff
        else:
            raise ValueError('gl is not 1 or 0')
        techsig[tech.isnull()] = np.nan        
        # I then multiply the previous iteration of sig (the first iteration is set above) by the just
        # created techsig dataframe. I multiply them together because that is equivalent to a logical
        # and. Therefore, I only get a 1 in the resulting sig dataframe if there is a 1 in each of the
        # other dataframes, and a 0 as long as there is one 0.
        sig = sig * techsig
    return sig

def get_measures(buy_chromosome,sell_chromosome,prices,closes,ma_keys,bol_keys,growth_keys,
            supmax_keys,supmin_keys,return_keys,pvma_keys,delta_context,price_context,pv_context):
    '''
    This function goes into fitness functions and returns a list with buy and sell technical dataframes,
    cutoffs, and gls (in that order with the three buy ones preceding the sell ones) according to the buy 
    and sell chromosomes, prices data, and contextualised super dataframes. For detail on the arguments
    look at the arguments in the fitness function description.
    '''
    # I start by defining empty lists that will aggregate the chromosome selected values.
    buy_technicals = []
    buy_cutoffs = []
    buy_gls = []
    sell_technicals = []
    sell_cutoffs = []
    sell_gls = []
    # I put these lists along with the buy and sell chromosomes into a guide variable to help condense the 
    # upcoming code.
    guide = [(buy_chromosome,buy_technicals,buy_cutoffs,buy_gls),
             (sell_chromosome,sell_technicals,sell_cutoffs,sell_gls)]

    # ----------------------------------- NON Z-SCORE MEASURES ---------------------------------------- #
    # The following measures are defined by each individual stocks' price/volume values irrelevant of
    # other values for stocks in the same industry. The way this works is that I use the buy and sell
    # chromosomes' indexes to check if a particular measure is in the chromosome, and if it is, I define
    # the measure using the values from the chromosome's 'params' column and append it to the buy or sell 
    # technical list, and append the cutoff and gl value to their corresponding list. Note that, 
    # mechanically, the for loop iterates over a pair of tuples, one for the buy values and one for the 
    # sell values. Note also that I shift all the measures by one at the end so that the value generated
    # at "yesterday's" close is available at "today's" open.
    
    # bol is the price in excess of a rolling mean, divided by a rolling standard deviation
    for chromosome,technical,cutoff,gl in guide:
        if 'bol' in chromosome.index:
            gene = chromosome['params'].loc['bol']
            tec = ((closes - closes.rolling(gene[0]).mean()) / closes.rolling(gene[0]).std()).shift(1)
            technical.append(tec)
            cutoff.append(chromosome['cutoffs'].loc['bol'])
            gl.append(chromosome['gls'].loc['bol'])

    # mas is a simple contextualised moving average difference of the close prices.
    for chromosome,technical,cutoff,gl in guide:
        if 'moving-average' in chromosome.index:
            gene = chromosome['params'].loc['moving-average']
            tec = ((closes.rolling(gene[0]).mean() - closes.rolling(gene[1]).mean()) / closes.rolling(
                gene[2]).mean()).shift(1)
            technical.append(tec)
            cutoff.append(chromosome['cutoffs'].loc['moving-average'])
            gl.append(chromosome['gls'].loc['moving-average'])

    # pv is a contextualised moving average difference of the product of close price and volume.        
    for chromosome,technical,cutoff,gl in guide: 
        if 'price-volume' in chromosome.index:
            gene = chromosome['params'].loc['price-volume']
            tec = closes * prices.xs('volume',level=1,axis=1)
            tec = ((tec.rolling(gene[0]).mean() - tec.rolling(gene[1]).mean()) / tec.rolling(
                gene[2]).mean()).shift(1)
            technical.append(tec)
            cutoff.append(chromosome['cutoffs'].loc['price-volume'])
            gl.append(chromosome['gls'].loc['price-volume'])

    # growth is just the return over some previous time period    
    for chromosome,technical,cutoff,gl in guide:
        if 'growth' in chromosome.index:
            gene = chromosome['params'].loc['growth']
            tec = ((closes / closes.shift(gene[0])) - 1).shift(1)
            technical.append(tec)
            cutoff.append(chromosome['cutoffs'].loc['growth'])
            gl.append(chromosome['gls'].loc['growth'])

    # supmin is the current price over the lowest point over some timeframe.
    for chromosome,technical,cutoff,gl in guide:
        if 'supmin' in chromosome.index:
            gene = chromosome['params'].loc['supmin']
            tec = (closes / prices.xs('low',level=1,axis=1).shift(1).rolling(gene[0]).min()).shift(1)
            technical.append(tec)
            cutoff.append(chromosome['cutoffs'].loc['supmin'])
            gl.append(chromosome['gls'].loc['supmin'])

    # supmax is the current price over the highest point over some timeframe        
    for chromosome,technical,cutoff,gl in guide:
        if 'supmax' in chromosome.index:
            gene = chromosome['params'].loc['supmax']
            tec = (closes / prices.xs('high',level=1,axis=1).shift(1).rolling(gene[0]).max()).shift(1)
            technical.append(tec)
            cutoff.append(chromosome['cutoffs'].loc['supmax'])
            gl.append(chromosome['gls'].loc['supmax'])

    # rr is a rolling average of some period returns. Note that 60 shift and 360 rolling mean seemed ok.        
    for chromosome,technical,cutoff,gl in guide:
        if 'rolling-returns' in chromosome.index:
            gene = chromosome['params'].loc['rolling-returns']
            tec = (closes / closes.shift(gene[0]))
            tec = tec.rolling(gene[1]).mean().shift(1)
            technical.append(tec)
            cutoff.append(chromosome['cutoffs'].loc['rolling-returns'])
            gl.append(chromosome['gls'].loc['rolling-returns'])

    # ------------------------------------------ Z-SCORE MEASURES ----------------------------------------- #
    # These measures are transformations of the dataframes of industry Z-scores. Given that I have to
    # define a raft of base Zscore dataframes beforehand in order to save time, I categorise the same
    # basic measure using a base Zscore dataframe with different parameters as a different measure.
    # Therefore, in this section I do the same as before, except that it is within the context of an
    # extra for loop that iterates checks over different permutations of a measure's name. These
    # permutations just have different numbers added to the end, indicating the parameter used in the
    # underlying dataframe. The number at the end of the permutation of the name is used to retrieve
    # the appropriate underlying dataframe from one of the super-dataframes that concatenates these
    # base dataframes calibrated with different parameters together.
    
    # The ma measures are the difference between two moving averages, without dividing by another moving
    # average, since we are already dealing with zscore values.
    for key in ma_keys:
        for chromosome,technical,cutoff,gl in guide:
            if key in chromosome.index:
                gene = chromosome['params'].loc[key]
                tec = price_context['price_context_' + key.split('_')[-1]]
                tec = (tec.rolling(gene[0]).mean() - tec.rolling(gene[1]).mean()).shift(1)
                technical.append(tec)
                cutoff.append(chromosome['cutoffs'].loc[key])
                gl.append(chromosome['gls'].loc[key])

    # The bol measures are like above
    for key in bol_keys:
        for chromosome,technical,cutoff,gl in guide:
            if key in chromosome.index:
                gene = chromosome['params'].loc[key]
                tec = price_context['price_context_' + key.split('_')[-1]]
                tec = ((tec - tec.rolling(gene[0]).mean()) / tec.rolling(gene[0]).std()).shift(1)
                technical.append(tec)
                cutoff.append(chromosome['cutoffs'].loc[key])
                gl.append(chromosome['gls'].loc[key])

    # The growth measures are obtained by subtraction rather than division, again because the values are
    # already Zscores. 
    for key in growth_keys:
        for chromosome,technical,cutoff,gl in guide:
            if key in chromosome.index:
                gene = chromosome['params'].loc[key]
                tec = price_context['price_context_' + key.split('_')[-1]]
                tec = (tec - tec.shift(gene[0])).shift(1)
                technical.append(tec)
                cutoff.append(chromosome['cutoffs'].loc[key])
                gl.append(chromosome['gls'].loc[key])

    # supmax measures are subtracted rather than divided because data is already Zscores. Note also that
    # I do not use highs or lows in the base Zscore dataframes.
    for key in supmax_keys:
        for chromosome,technical,cutoff,gl in guide:
            if key in chromosome.index:
                gene = chromosome['params'].loc[key]
                tec = price_context['price_context_' + key.split('_')[-1]]
                tec = (tec - tec.shift(1).rolling(gene[0]).max()).shift(1)
                technical.append(tec)
                cutoff.append(chromosome['cutoffs'].loc[key])
                gl.append(chromosome['gls'].loc[key])

    # supmin measures are also subtracted and do not use highs or lows in the base dataframes.
    for key in supmin_keys:
        for chromosome,technical,cutoff,gl in guide:
            if key in chromosome.index:
                gene = chromosome['params'].loc[key]
                tec = price_context['price_context_' + key.split('_')[-1]]
                tec = (tec - tec.shift(1).rolling(gene[0]).min()).shift(1)
                technical.append(tec)
                cutoff.append(chromosome['cutoffs'].loc[key])
                gl.append(chromosome['gls'].loc[key])

    # The returns measures just take a rolling mean of the underlying dataframes.
    for key in return_keys:
        for chromosome,technical,cutoff,gl in guide:
            if key in chromosome.index:
                gene = chromosome['params'].loc[key]
                tec = delta_context['delta_context_' + key.split('_')[-1]]
                tec = tec.rolling(gene[0]).mean().shift(1)
                technical.append(tec)
                cutoff.append(chromosome['cutoffs'].loc[key])
                gl.append(chromosome['gls'].loc[key])

    # pvma is a moving average difference for price-volume Zscore dataframes.
    for key in pvma_keys:
        for chromosome,technical,cutoff,gl in guide:
            if key in chromosome.index:
                gene = chromosome['params'].loc[key]
                tec = pv_context['pv_context_' + key.split('_')[-1]]
                tec = (tec.rolling(gene[0]).mean() - tec.rolling(gene[1]).mean()).shift(1)
                technical.append(tec)
                cutoff.append(chromosome['cutoffs'].loc[key])
                gl.append(chromosome['gls'].loc[key])

    # -------------------------------------- STRAIGHT FROM SUPER-DATAFRAMES -------------------------------- #
    # The following measures are just the individual Zscore dataframes that were used to create the
    # measures above. I only do so for the price_context and pv_context mega dataframes because the 
    # growth_context dataframe can be obtained by optimising the parameter to one. 
    
    for key in price_context.columns.levels[0]:
        for chromosome,technical,cutoff,gl in guide:
            if key in chromosome.index:
                technical.append(price_context[key].shift(1))
                cutoff.append(chromosome['cutoffs'].loc[key])
                gl.append(chromosome['gls'].loc[key])

    for key in pv_context.columns.levels[0]:
        for chromosome,technical,cutoff,gl in guide:
            if key in chromosome.index:
                technical.append(pv_context[key].shift(1))
                cutoff.append(chromosome['cutoffs'].loc[key])
                gl.append(chromosome['gls'].loc[key])
    # I return a list as explained in the function description.
    return [buy_technicals,buy_cutoffs,buy_gls,sell_technicals,sell_cutoffs,sell_gls]

def action_maker(signals):
    '''
    action_maker takes a signals dataframe and turns it into a dictionary of actions. 
    '''
    # I then stringarise the signals dataframe and use the buy_dates and sell_dates function to get a
    # series with a list of buy/sell dates for each stock.
    buys = signals.apply(stringarise).apply(lambda x: buy_dates(x,signals.index))
    sells = signals.apply(stringarise).apply(lambda x: sell_dates(x,signals.index))

    # I turn the buys and sells series into a combined list of tuples with '_b' or _'s' added 
    # to the stonk ticker to indicate a buy or sell action in the 1th iloc of the tuple.
    lol = [[(stonk, tup[1] + '_b') for stonk in tup[0]] for tup in zip(buys,buys.index)] + \
    [[(stonk, tup[1] + '_s') for stonk in tup[0]] for tup in zip(sells,sells.index)]
    flat_bs = [tup for sub in lol for tup in sub]
    # I gather all the unique dates from the 0th iloc of the tuples in the list into the dates set.
    dates = pd.Series(list(set([tup[0] for tup in flat_bs]))).sort_values()
    # The actions dictionary takes each date in dates, finds all the tuples in flat_bs with that date and
    # '_b' stonks and puts them in the 'buy' kry of that date's subdictionary. It also finds all the
    # tuples with that date and '_s' stonks and puts them in the 'sell' key of the sub dataframe. The
    # dictionary is the cornerstone of the fitness function and is used to iterate through action dates
    # and determine the actions that take place.
    return {date:{'buy':[tup[1][:-2] for tup in flat_bs if tup[0] == date and tup[1][-2:] == '_b'],
              'sell':[tup[1][:-2] for tup in flat_bs if tup[0] == date and tup[1][-2:] == '_s']} 
              for date in dates}
              
def keepers(series,max_limit=5,proportion_limit=0.1):
    '''
    droppers identifies, from the first valid index, sparse series. It returns True if the maximum
    consecutive number of nans exceeds the max_limit, or if the proportion of nans is greater than the
    proportion_limit.
    '''
    if sum(series) == len(series):
        return np.nan
    # I find the first valid index and then stringarise the series after that index into a string
    # of ones and zeros.
    start = drop_true(series).index[0]
    isnul = series[start:]
    string = str(isnul.apply(lambda x: 1 if x else 0).tolist()).replace('[','').replace(
    ']','').replace(',','').replace(' ','')
    # Then I use re's findall function to return a list of all the consecutive 1s and take the len of 
    # the consecutive 1s strings so that I am left with a list of numbers representing consecutive nan
    # lengths.
    consecs = [len(match) for match in re.findall('1+',string)]
    # I then check if the maximum consecutive nan length or nan proportion is greater than given 
    # cuttoffs.
    if robust_max(consecs) >= max_limit or (sum(isnul) / len(isnul)) >= proportion_limit:
        return np.nan
    else:
        return False
        
def OG_signal(buy_technicals,buy_cutoffs,buy_gls,sell_technicals,sell_cutoffs,sell_gls):
    '''
    This function takes the technicals dataframes, cutoffs, and gls, and uses them to produce a signals
    dataframe that has a 1 for a buy signal and -1 for a sell signal. 0 is a null signal.
    '''
    # I create a buy and sell signal from the technicals, cutoffs, and gls, using the signal_maker function, 
    # which returns ones for a go signal and zeros for a null signal. Subtracting sell from buy gives me the
    # original signals dataframe
    buy_signal = signal_maker(buy_technicals,buy_cutoffs,buy_gls)
    sell_signal = signal_maker(sell_technicals,sell_cutoffs,sell_gls)
    return buy_signal - sell_signal
    
def refiner(signals,action_prices,max_limit=5,proportion_limit=0.1):
    '''
    This function takes an OG signal and action_prices and refines them by removing those stocks where 
    signals or prices are too sparse and removing dates where there is no buy or sell signal. Finally,
    the function makes sure there are nans in the signals dataframe where there is no corresponding price
    data. Note that it returns a tuple of the signals dataframe and a refined action_prices dataframe. 
    Look at the 'keepers' function for more details on how I refine the stocks.
    '''
    # I start by creating a filter that identifies those dates in which there is at least one signal and
    # refine both the signals and action_prices dataframe so that they contain only such dates. Note that
    # I add a .copy() to avoid a splice warning.
    signal_filter = (abs(signals) == 1).sum(axis=1) > 0
    signals = signals[signal_filter].copy()
    action_prices = action_prices[signal_filter]
    # Then I identify the nans in the action_prices dataframe and use them as a mask to ensure that where
    # there is a price nan, there is also a signal nan.
    nulprice = action_prices.isnull()
    signals[nulprice] = np.nan
    # I then apply keepers to the signals isna boolean array to identify the stocks worth keeping
    nulsig = signals.isnull()
    keeps = nulsig.apply(lambda x: keepers(x,max_limit,proportion_limit)).dropna().index
    # Finally I return the signals and action_prices dataframe with only the keepers stocks.
    return signals[keeps], action_prices[keeps]

def simulation(actions,action_prices,start,num_stocks,min_invest,spread,first_signal):
    '''
    simulation takes an actions dictionary and other arguments to run a simulaltion where I buy and sell
    stocks. The simulation returns the daily compound return calculated by taking the total return
    and rooting to the order of the time delta between the first signal and the last price. All the
    arguments are explained in the fitness function, except for actions, which is a dictionary detailing
    buy and sell actions at different dates, and first_signal, which is the timestamp representing the 
    first valid signal date.
    '''
    
    available = start
    portfolio = {}
    # I start a for loop with the dates in order.
    for date in actions:
        # buy is a list of stocks that the signals tell me to buy, and active_sell is the stocks for
        # which there are sell signals and that are already in the portfolio.
        buy = actions[date]['buy']
        sell = actions[date]['sell']
        active_sell = [stonk for stonk in sell if stonk in portfolio.keys()]
        # If active_sell has some stonks, then sell action takes place...
        if len(active_sell) > 0:
            # ...For each stonk in active sells, I find the selling price and divide it by the purchase
            # price of the stonk as specified in the portfolio. I multiply this return by the amount
            # invested in that stonk. I sum over the sold stonks and add it to the available amount.
            available = available + sum([(action_prices.at[date,stonk] / portfolio[stonk]['price']) * 
                                         portfolio[stonk]['amount'] for stonk in active_sell])
            # Then I delete the stonks in the portfolio.
            for key in active_sell:
                del portfolio[key]
        else:
            pass
        # Then I define the 'space' variable, which indicates the number of stocks that can be invested
        # in at that moment. It is the minimum of [the maximum number of stocks to invest in minus the
        # current stocks in the portfolio] and [the integer value of the available amount divided by the
        # minimum amount I can invest]. The second argument of this minimum is so that I never invest
        # more than the minimum amount into a stock.
        space = min(num_stocks - len(portfolio.keys()), int(available / min_invest))        
        # Then, if there are any buy signals, and there is space I proceed - note that if I had less than
        # the minimum amount you can invest, then 'space' would be 0 and I would not proceed.
        if len(buy) > 0 and space > 0:
            # I define the amount I will put into each stock as the available amount divided by the space
            # variable. This ensures that when I have the maximum number of stocks in my portfolio, I
            # will have spent all my money. It also ensures I never invest more than the minimum amount
            # because of how space is defined.
            per_stonk = available / space
            # If there are more stocks to buy than space available, I randomly choose stocks to buy.
            buy = random.sample(buy,k=min(space,len(buy)))
            # I then update the portfolio with the stocks I bought, their purchase price including
            # spread, and the amount invested in each.
            portfolio.update({stonk:{'amount':per_stonk,'price':(1+spread) * action_prices.at[date,
                stonk]} for stonk in buy})
            # I then subtract from available the total cost of the stock purchase.
            available = available - (per_stonk * len(buy))
        else:
            pass
    # After the whole for loop, sum the available with the value of the portfolio as given by the last
    # available price for each stock. These last available prices cannot be for dates that are too far
    # away from each other due to my dropping stocks with sparse data earlier.
    total = available + sum([(action_prices.at[action_prices[stonk].last_valid_index(),stonk] / portfolio[stonk]['price']) * portfolio[
                stonk]['amount'] for stonk in portfolio])
    # Finally, I return the total return as a daily return.
    return (total / start) ** (1 / (action_prices.last_valid_index() - first_signal).days)
    
def diagnostic_simulation(actions,action_prices,start,num_stocks,min_invest,spread,first_signal):
    '''
    This function is just like the simulation function except that it returns a triple of the portfolio
    dictionary over time, the portfolio value over time, and the final value. It can be used to graph
    performance and track stock purchases.
    '''
    dynamic_portfolio = {}
    dynamic_total = {}
    available = start
    portfolio = {}
    # I start a for loop with the dates in order.
    for date in list(actions.keys()):
        # buy is a list of stocks that the signals tell me to buy, and active_sell is the stocks for
        # which there are sell signals and that are already in the portfolio.
        buy = actions[date]['buy']
        sell = actions[date]['sell']
        active_sell = [stonk for stonk in sell if stonk in portfolio.keys()]
        # If active_sell has some stonks, then sell action takes place...
        if len(active_sell) > 0:
            # ...For each stonk in active sells, I find the selling price and divide it by the purchase
            # price of the stonk as specified in the portfolio. I multiply this return by the amount
            # invested in that stonk. I sum over the sold stonks and add it to the available amount.
            available = available + sum([(action_prices.at[date,stonk] / portfolio[stonk]['price']) * 
                                         portfolio[stonk]['amount'] for stonk in active_sell])
            # Then I delete the stonks in the portfolio.
            for key in active_sell:
                del portfolio[key]
        else:
            pass
        # Then I define the 'space' variable, which indicates the number of stocks that can be invested
        # in at that moment. It is the minimum of [the maximum number of stocks to invest in minus the
        # current stocks in the portfolio] and [the integer value of the available amount divided by the
        # minimum amount I can invest]. The second argument of this minimum is so that I never invest
        # more than the minimum amount into a stock.
        space = min(num_stocks - len(portfolio.keys()), int(available / min_invest))        
        # Then, if there are any buy signals, and there is space I proceed - note that if I had less than
        # the minimum amount you can invest, then 'space' would be 0 and I would not proceed.
        if len(buy) > 0 and space > 0:
            # I define the amount I will put into each stock as the available amount divided by the space
            # variable. This ensures that when I have the maximum number of stocks in my portfolio, I
            # will have spent all my money. It also ensures I never invest more than the minimum amount
            # because of how space is defined.
            per_stonk = available / space
            # If there are more stocks to buy than space available, I randomly choose stocks to buy.
            buy = random.sample(buy,k=min(space,len(buy)))
            # I then update the portfolio with the stocks I bought, their purchase price including
            # spread, and the amount invested in each.
            portfolio.update({stonk:{'amount':per_stonk,'price':(1+spread) * action_prices.at[date,stonk]
                                    } for stonk in buy})
            # I then subtract from available the total cost of the stock purchase.
            available = available - (per_stonk * len(buy))
        else:
            pass
         # The last action at each iteration of the for loop is to update dynamic_portfolio, calculate the
         # current portfolio total, and update that to dynamic_total
        dynamic_portfolio.update({date:portfolio.copy()})
        total = available + sum([(action_prices.at[date,stonk] / portfolio[stonk]['price']) * portfolio[
                stonk]['amount'] for stonk in portfolio])
        dynamic_total.update({date:total})
    # After the whole for loop, sum the available with the value of the portfolio as given by the last
    # available price for each stock. These last available prices cannot be for dates that are too far
    # away from each other due to my dropping stocks with sparse data earlier.
    final_total = available + sum([(action_prices.at[action_prices[stonk].last_valid_index(),
                                    stonk] / portfolio[stonk]['price']) * portfolio[stonk][
                                    'amount'] for stonk in portfolio])
    # Finally, I return the tripe.
    return (dynamic_portfolio,dynamic_total,final_total)
    
def final_fitness(chromosome,prices,closes,action_prices,ma_keys,bol_keys,growth_keys,
            supmax_keys,supmin_keys,return_keys,pvma_keys,delta_context,price_context,pv_context,
            min_density=0.5,spread=0.09/100,min_invest=50,start=5000,max_limit=5,proportion_limit=0.1):
            
    '''
    This is like the standard fitness function except that it returns the scalers and columns of the 
    technicals dataframes, so that out of sample data can be transformed and ultimately tested.
    '''
    
    # I split the chromosome into its constituent components.
    buy_chromosome, sell_chromosome, num_stocks = chromosome
    # I use the get_measures function to get the lists of technical dataframes, cutoffs, and gls.
    buy_technicals,buy_cutoffs,buy_gls,sell_technicals,sell_cutoffs,sell_gls = get_measures(
        buy_chromosome,sell_chromosome,prices,closes,ma_keys,bol_keys,growth_keys,supmax_keys,
        supmin_keys,return_keys,pvma_keys,delta_context,price_context,pv_context)
    
    # Supressing some useless warnings, I standardise the technicals dataframes. This time, I keep 
    # the fitted standard scaler objects as well as each technicals' dataframes' columns. These
    # will be returned along with the score so that I can scale test data properly.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            buy_scalers = [StandardScaler().fit(tech) for tech in buy_technicals]
            buy_columns = [tech.columns for tech in buy_technicals]
            buy_technicals = [pd.DataFrame(buy_scalers[i].transform(buy_technicals[i]),
                                           index=buy_technicals[i].index,
                                           columns=buy_columns[i]) 
                                           for i in range(len(buy_scalers))] 
            sell_scalers = [StandardScaler().fit(tech) for tech in sell_technicals]
            sell_columns = [tech.columns for tech in sell_technicals]
            sell_technicals = [pd.DataFrame(sell_scalers[i].transform(sell_technicals[i]),
                                           index=sell_technicals[i].index,
                                           columns=sell_columns[i]) 
                                           for i in range(len(sell_scalers))]
        # I catch any value errors due to division by near zeros in technical dataframes.
        except ValueError:
            return (np.nan,np.nan,np.nan,np.nan,np.nan)
    # Using the resulting technicals, I use OG_signal to produce a signals dataframe. I also
    # set the first valid index of the signals dataframes as first_signal.
    signals = OG_signal(buy_technicals,buy_cutoffs,buy_gls,sell_technicals,sell_cutoffs,sell_gls)
    first_signal = signals.first_valid_index()

    # If there are no buy signals, or there are too few signals relative to prices, or there are 
    # fewer than zero days between the first valid signal and the last available price date, then 
    # I return np.nan.
    if (len(signals[first_signal:]) / len(action_prices) < min_density or (signals==1).sum().sum() == 0 
        or (action_prices.last_valid_index() - first_signal).days <= 0):
        return (np.nan, np.nan, np.nan, np.nan, np.nan)
    # I then refine the signals and action_prices dataframes to include only relevant information.
    signals,action_prices = refiner(signals, action_prices, max_limit, proportion_limit)
    # Then I create the actions dictionary with action_maker...
    actions = action_maker(signals)
    # ...and run the simulation.
    score = simulation(actions, action_prices, start, num_stocks, min_invest, spread, first_signal)
    # finally, I return the score, buy and sell scaler objects, and buy and sell columns.
    return (score, buy_scalers, buy_columns, sell_scalers, sell_columns)
    
def fitness(chromosome,prices,closes,action_prices,ma_keys,bol_keys,growth_keys,
            supmax_keys,supmin_keys,return_keys,pvma_keys,delta_context,price_context,pv_context,
            min_density=0.5,spread=0.09/100,min_invest=50,start=5000,max_limit=5,proportion_limit=0.1):
    '''
    This is the fitness function that returns the daily compound return in 1 + x form, owing to the
    strategy given by the buy and sell chromosomes, and the maximum number of stocks to invest in in any
    one time. The arguments are:
    - buy_chromosome: This is a dataframe containing a params column wich details the parameters that
    make the measures that form the signal, a cutoffs column that defines the cutoff in terms of Zscore
    for each measure, and a gl column detailing whether the signal is positive when the measure is greater
    than or less than the given cutoff. The dataframe's index is the name of the measures being employed.
    - sell_chromosome: just like the buy_chromosome but for the sell signal.
    - prices: This is the dataframe with high,low,open,close,and volume values.
    - closes: This is a dataframe of close prices. I have it as an argument to save having to define it 
    from prices within the function each time.
    - action_prices: This is the dataframe of prices that will actually be used for buying and selling.
    Currently, At first, I had it as the open prices. It may still be that, depending on when I am 
    reading this.
    - _keys arguments: These arguments contain the names of the measures created with different 
    calibrations of the base industry Zscore dataframes represented in the following arguments.
    - _context arguments: These arguments contain the base mega-dataframes with different calibrations
    of industry Zscore dataframes.
    - num_stocks: This defines the maximum number of stocks that will be invested in in any one time.
    - min_density: This defines the minimum proportion of dates for which there must be some signal for
    some stock, as a proportion of the total number of dates for which there is price data.
    - spread: This is the proportion of a stock's price that will be added to its buying price.
    - min_invest: This is the minimum amount that you can put in a stock.
    - start: This is the initial amount you have to invest.
    '''
    
    # I split the chromosome into its constituent components.
    buy_chromosome, sell_chromosome, num_stocks = chromosome
    # I use the get_measures function to get the lists of technical dataframes, cutoffs, and gls.
    buy_technicals,buy_cutoffs,buy_gls,sell_technicals,sell_cutoffs,sell_gls = get_measures(
        buy_chromosome,sell_chromosome,prices,closes,ma_keys,bol_keys,growth_keys,supmax_keys,
        supmin_keys,return_keys,pvma_keys,delta_context,price_context,pv_context)
    # Supressing some useless warnings, I standardise the technicals dataframes.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            buy_technicals = [pd.DataFrame(StandardScaler().fit_transform(tech),index=tech.index,
                                           columns=tech.columns) 
                                           for tech in buy_technicals] 
            sell_technicals = [pd.DataFrame(StandardScaler().fit_transform(tech),index=tech.index,
                                            columns=tech.columns) 
                                            for tech in sell_technicals]
        # I catch any value errors due to division by near zeros in technical dataframes.
        except ValueError:
            return np.nan
    # Using the resulting technicals, I use OG_signal to produce a signals dataframe. I also
    # set the first valid index of the signals dataframes as first_signal.
    signals = OG_signal(buy_technicals,buy_cutoffs,buy_gls,sell_technicals,sell_cutoffs,sell_gls)
    first_signal = signals.first_valid_index()

    # If there are no buy signals, or there are too few signals relative to prices, then 
    # I return np.nan.
    if len(signals[first_signal:]) / len(action_prices) < min_density or (signals==1).sum().sum() == 0:
        return np.nan
    # I then refine the signals and action_prices dataframes to include only relevant information.
    signals,action_prices = refiner(signals, action_prices, max_limit, proportion_limit)
    # I make a final check that there aren't fewer than zero days between the first valid signal 
    # and the last available price date.
    if (action_prices.last_valid_index() - first_signal).days <= 0:
        return np.nan
    
    # Then I create the actions dictionary with action_maker...
    actions = action_maker(signals)
    # ...and run the simulation, returning its result.
    return simulation(actions, action_prices, start, num_stocks, min_invest, spread, first_signal)
    
def diagnostic_fitness(chromosome,prices,closes,action_prices,ma_keys,bol_keys,growth_keys,
            supmax_keys,supmin_keys,return_keys,pvma_keys,delta_context,price_context,pv_context,
            min_density=0.5,spread=0.09/100,min_invest=50,start=5000,max_limit=5,proportion_limit=0.1):
      
    '''
    This is just like the fitness function except that it returns the result of diagnostic_simulation
    rather than that of simulation. Use this on a potential solution to then be able to draw a graph of
    its performance.
    '''
    
    # I split the chromosome into its constituent components.
    buy_chromosome, sell_chromosome, num_stocks = chromosome
    # I use the get_measures function to get the lists of technical dataframes, cutoffs, and gls.
    buy_technicals,buy_cutoffs,buy_gls,sell_technicals,sell_cutoffs,sell_gls = get_measures(
        buy_chromosome,sell_chromosome,prices,closes,ma_keys,bol_keys,growth_keys,supmax_keys,
        supmin_keys,return_keys,pvma_keys,delta_context,price_context,pv_context)
    # Supressing some useless warnings, I standardise the technicals dataframes.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            buy_technicals = [pd.DataFrame(StandardScaler().fit_transform(tech),index=tech.index,
                                           columns=tech.columns) 
                                           for tech in buy_technicals] 
            sell_technicals = [pd.DataFrame(StandardScaler().fit_transform(tech),index=tech.index,
                                            columns=tech.columns) 
                                            for tech in sell_technicals]
        # I catch any value errors due to division by near zeros in technical dataframes.
        except ValueError:
            return np.nan
    # Using the resulting technicals, I use OG_signal to produce a signals dataframe. I also
    # set the first valid index of the signals dataframes as first_signal.
    signals = OG_signal(buy_technicals,buy_cutoffs,buy_gls,sell_technicals,sell_cutoffs,sell_gls)
    first_signal = signals.first_valid_index()

    # If there are no buy signals, or there are too few signals relative to prices, then 
    # I return np.nan.
    if len(signals[first_signal:]) / len(action_prices) < min_density or (signals==1).sum().sum() == 0:
        return np.nan
    # I then refine the signals and action_prices dataframes to include only relevant information.
    signals,action_prices = refiner(signals, action_prices, max_limit, proportion_limit)
    # I make a final check that or there aren't fewer than zero days between the first valid signal 
    # and the last available price date.
    if (action_prices.last_valid_index() - first_signal).days <= 0:
        return np.nan
    
    # Then I create the actions dictionary with action_maker...
    actions = action_maker(signals)
    # ...and run the diagnostic_simulation, returning its result.
    return diagnostic_simulation(actions, action_prices, start, num_stocks, min_invest, spread, first_signal)

def population_count(buy_seed,sell_seed,multiplier=5,min_pop=8,max_pop=80):
    '''
    This function returns multiplier times the total length of the seeds but is limited by min_pop
    and max_pop.
    '''
    return min(max((len(buy_seed) + len(sell_seed)) * multiplier, min_pop), max_pop)

def starter_population(buy_seed,sell_seed,pop_count,max_stock=150):
    '''
    This function creates an initial population of solutions as a list of triplets with length of 
    pop_count. the first of each triplet is the buy_chromosome, the second is the sell_chromosome, and 
    the last is the maximum number of stocks that can be in the portfolio, which is limited by max_stock. 
    '''
    # I assign appropriate indexes
    buy_index, sell_index = buy_seed.index, sell_seed.index
    # I then return a list of tuples of buy_chromosome, sell_chromosome, and maximum stock number.
    return [(
    pd.DataFrame([buy_seed.apply(lambda x: np.random.randint(1,101,size=x)).tolist(),
                  np.random.normal(size=len(buy_index)),
                  np.random.randint(2,size=len(buy_index))],
                  columns=buy_index,
                  index=['params','cutoffs','gls']).T,
    pd.DataFrame([sell_seed.apply(lambda x: np.random.randint(1,101,size=x)).tolist(),
                  np.random.normal(size=len(sell_index)),
                  np.random.randint(2,size=len(sell_index))],
                  columns=sell_index,
                  index=['params','cutoffs','gls']).T,
    np.random.randint(1,max_stock)
    ) for i in range(pop_count)]

def selection(n,target_length,child_proportion):
    '''
    This function returns a list of pairs of numbers. The pair of numbers represent a pair of chromosomes
    that will have sex to produce a set of offspring. The numbers themselves represent the rank of the
    selected chromosomes (high number equals high rank) which can be used to actually select the
    chromosomes in the mating process. The function requires the current population size *n*,
    target_length, and child_proportion. target_length is the original target size of the population,
    and it is used along with child_proportion £ (0,1) to determine how many unique pairs are to be
    selected. If there are not enough unique pairs to fill this requirement then I return a nan because
    it is not worth continuing.
    '''
    # choice_length is the integer value of the product of target_length and child proportion. This way,
    # by choosing choice_length pairs to mate, we get roughly target_length * child_proportion new
    # children, since each pair produced two offspring.
    choice_length = int((target_length*child_proportion) / 2)
    # combs is all the possible combinations expressed as a touple of rank-numbers.
    combs = list(combinations(np.arange(n),2))
    # If there aren't enough combinations to create choice_length new children, then the population has
    # degenerated to the extent that I should stop the GA, so I return np.nan.
    if choice_length > len(combs):
        return None
    # combrank is the sum of the ranks in the combs' pairs. dividing each of these sums by the total of
    # the sums gives the probability of selecting each unique pair. The total is given by the formula
    # shown, as I have derived and tested.
    combrank = [tup[0] + tup[1] + 2 for tup in combs]
    probs = np.array(combrank) / (n*(n+1)*((2*n-2)/4))
    # I then sample without replacement amongst the combs with the probability defined above. Note that,
    # mechanically, I sample the indexes and then refine the combs list.
    choices = np.random.choice(len(combs),size=choice_length,replace=False,p=probs)
    choices = [combs[i] for i in choices]
    return choices
    
def int_sex(tup,overhang=0.15):
    '''
    int_sex is a function that combines integers by using a linear combination. The lambda used to
    combine the integers is between -overhand and 1 + overhang, and the resulting numbers produced are
    rounded.
    '''
    lincomb = np.random.uniform(-overhang,1 + overhang)
    num1,num2 = tup[0],tup[1]
    crossed=[round(lincomb * num1 + (1-lincomb) * num2), round((1 - lincomb) * num1 + (lincomb) * num2)]
    return [child + 1 if child == 0 else child for child in crossed]
    
def binary_sex(couple):
    '''
    binary sex takes a pair of numbers in a tuple and combines them by binarising them and crossing them
    over. It returns a pair of numbers corresponding to the two resulting crossover numbers produced by
    taking one direction of crossover versus another.
    '''
    # I set the numbers in the tuple as seperate variables.
    num1,num2 = couple[0],couple[1]
    # I assign a string of zeros to a bunch of variables to help homogonise the length of the numbers.
    zeros = '00000000'
    # I set a list of the numbers as a binary string without '0b' and I define the maximum length amongst
    # the two binary strings.
    binary = [bin(num1).split('b')[-1],bin(num2).split('b')[-1]]
    max_len = max([len(num) for num in binary])
    # If one of the binary numbers is shorter than the others, I add zeros to it until the binary numbers
    # are the same length.
    binary = [zeros[:max_len-len(num)] + num for num in binary]
    # I define a random integer crossover point.
    cross = np.random.randint(max_len)
    # I then create the two potential combinations from the crossover as a list of a pair of numbers
    children =  [int(binary[0][:cross] + binary[1][cross:],2), 
                 int(binary[1][:cross] + binary[0][cross:],2)]
    # And finally, I add a one to any solution that is equal to zero, since zero is an invalid solution.
    return [child + 1 if child == 0 else child for child in children]
    
def binary_mutant_sex(couple,mutation_rate):
    '''
    binary_mutant_sex is like binary_sex, except that it adds mutations with mutation_rate probability.
    The mutations work by creating a list of ones zeros, with ones appearing at mutation_rate frequency.
    Once the children binary numbers have been created, they are added into a single string and, if there
    is a one in the mutations list for the corresponding position in the string, the bit is flipped,
    resulting in the mutant children.
    '''
    # I set the numbers in the couple as seperate variables.
    num1,num2 = couple[0],couple[1]
    # I assign a string of zeros to a bunch of variables to help homogonise the length of the numbers.
    zeros = '00000000'
    # I set a list of the numbers as a binary string without '0b' and I define the maximum length amongst
    # the two binary strings.
    binary = [bin(num1).split('b')[-1],bin(num2).split('b')[-1]]
    max_len = max([len(num) for num in binary])
    # If one of the binary numbers is shorter than the others, I add zeros to it until the binary numbers
    # are the same length.
    binary = [zeros[:max_len-len(num)] + num for num in binary]
    # I define a random integer crossover point.
    cross = np.random.randint(max_len)
    # I then create the two children using the crossover.
    children =  [binary[0][:cross] + binary[1][cross:], binary[1][:cross] + binary[0][cross:]]
    # I combine  the strings of the two children and transform it into a list of numbers instead of a
    # string so that I can operate on them.
    full_chromo = [int(bit) for bit in children[0] + children[1]]
    # I create a mutations list of ones and zeros of the same length as full_chromo to direct the
    # mutations
    mutations = np.random.choice([1,0],size=len(full_chromo),p=[mutation_rate,1-mutation_rate])
    # I flip the bit if there is a one in mutations and leave it otherwise
    mutated = [1-full_chromo[i] if mutations[i] == 1 else full_chromo[i] for i in range(len(full_chromo))]
    # I use .join to transform the list back into a string.
    mutated_string = "".join([str(bit) for bit in mutated])
    # I split the string back into its children strings and transform them into decimal integers.
    mutated_children = [int(mutated_string[:max_len],2),int(mutated_string[max_len:],2)]
    # And finally, I add a one to any solution that is equal to zero, since zero is an invalid solution.
    return [child + 1 if child == 0 else child for child in mutated_children]
    
def normal_sex(couple,compressor=4):
    '''
    This function returns a pair of normally distributed random numbers centered on the mean of the
    tuple and with a standard deviation equal to the absolute difference between the numbers divided by
    compressor.
    '''
    num1, num2 = couple[0], couple[1]
    mean = (num1 + num2) / 2
    std = abs(num1 - num2) / compressor
    return [np.random.normal(mean,std), np.random.normal(mean,std)]
    
def normal_mutant_sex(couple,compressor=4,mutation_rate=0.015):
    '''
    This function is like normal_sex except that it perterbs the children with an additional normal shock
    with a probability equal to mutation_rate for each child.
    '''
    num1, num2 = couple[0], couple[1]
    mean = (num1 + num2) / 2
    std = abs(num1 - num2) / compressor
    children = [np.random.normal(mean,std), np.random.normal(mean,std)]
    mutations = np.random.choice([1,0],size=2,p=[mutation_rate,1-mutation_rate])
    return [np.random.normal(children[i],std) if mutations[i] == 1 else children[i] for i in range(2)]
    
def chromosex(m,f,compressor=4):
    '''
    chromosex takes a male and female chromosome and returns a list of two children chromosomes. The sort 
    of chromosome inputted must be a buy or sell chromosome dataframe. 
    '''
    # The first stage is to breed the integer parameters of the male and female chromosomes. I zip the 
    # 'param' columns of the male and female chromosomes.
    param_flirt = zip(m['params'],f['params'])
    # I then breed the numbers in the params columns using binary_sex.
    new_params = [[binary_sex(tup) for tup in zip(sol[0],sol[1])] for sol in param_flirt]
    # Finally, I seperate out the children solutions into two new columns, one for each child chromosome.
    b_param, g_param = [[[sol[i] for sol in pair] for pair in new_params] for i in range(2)]
    
    # The next step is to breed the cutoff points using normal_sex. Again, I zip the columns of the 
    # parent chromosomes together.
    cutoff_flirt = zip(m['cutoffs'],f['cutoffs'])
    # Then I bread each pair.
    new_cutoffs = [normal_sex(tup,compressor) for tup in cutoff_flirt]
    # And seperate them into two columns.
    b_cutoff, g_cutoff = [[pair[i] for pair in new_cutoffs] for i in range(2)]
    
    # The last step is to breed the gls. I zip the gls columns of the parent chromosomes together.
    gl_flirt = list(zip(m['gls'],f['gls']))
    # Then I breed the gls by randomly selecting either one or the other gl, twice, to produce two
    # randomly mixed gl columns.
    b_gl,g_gl = [[np.random.choice([tup[0],tup[1]]) for tup in gl_flirt] for i in range(2)]
    
    # I put together the three pairs of columns just created together to produce two new chromosomes.
    return (pd.DataFrame([b_param,b_cutoff,b_gl],columns=m.index,index=['params','cutoffs','gls']).T,
            pd.DataFrame([g_param,g_cutoff,g_gl],columns=f.index,index=['params','cutoffs','gls']).T)
            
def mutant_chromosex(m,f,compressor,binary_rate,normal_rate,gl_rate):
    '''
    This is like chromosex except that it uses mutant versions of the sex functions so as to breed two chromosomes
    with mutations. binary_rate is the mutation rate used in binary mutant sex functions and normal rate is the
    rate used in normal mutant sex functions.
    '''
    # The first stage is to breed the integer parameters of the male and female chromosomes. I zip the 
    # 'param' columns of the male and female chromosomes.
    param_flirt = zip(m['params'],f['params'])
    # I then breed the numbers in the params columns using binary_mutant_sex. 
    new_params = [[binary_mutant_sex(tup,binary_rate) for tup in zip(sol[0],sol[1])] for 
                  sol in param_flirt]
    # Finally, I seperate out the children solutions into two new columns, one for each child chromosome.
    b_param, g_param = [[[sol[i] for sol in pair] for pair in new_params] for i in range(2)]
    
    # The next step is to breed the cutoff points using normal_mutant_sex. Again, I zip the columns of 
    # the parent chromosomes together.
    cutoff_flirt = zip(m['cutoffs'],f['cutoffs'])
    # Then I bread each pair.
    new_cutoffs = [normal_mutant_sex(tup,compressor,normal_rate) for tup in cutoff_flirt]
    # And seperate them into two columns.
    b_cutoff, g_cutoff = [[pair[i] for pair in new_cutoffs] for i in range(2)]
    
    # The last step is to breed the gls. I zip the gls columns of the parent chromosomes together.
    gl_flirt = list(zip(m['gls'],f['gls']))
    # Then I breed the gls by randomly selecting either one or the other gl, twice, to produce two
    # randomly mixed gl columns.
    proto_children = [[np.random.choice([tup[0],tup[1]]) for tup in gl_flirt] for i in range(2)]
    mutations = [np.random.choice([1,0],p=[gl_rate,1-gl_rate],size=[len(child) for child in proto_children][i]) for i in range(2)]
    b_gl,g_gl = [[1 - proto_children[i][k] if mutations[i][k] == 1 else proto_children[i][k] for k in 
                  range(len(proto_children[i]))] for i in range(2)]
     
    # I put together the three pairs of columns just created together to produce two new chromosomes.
    return (pd.DataFrame([b_param,b_cutoff,b_gl],columns=m.index,index=['params','cutoffs','gls']).T,
            pd.DataFrame([g_param,g_cutoff,g_gl],columns=f.index,index=['params','cutoffs','gls']).T)
            
def old_mutant_chromosex(m,f,compressor,binary_rate,normal_rate):
    '''
    This is like chromosex except that it uses mutant versions of the sex functions so as to breed two chromosomes
    with mutations. binary_rate is the mutation rate used in binary mutant sex functions and normal rate is the
    rate used in normal mutant sex functions.
    '''
    # The first stage is to breed the integer parameters of the male and female chromosomes. I zip the 
    # 'param' columns of the male and female chromosomes.
    param_flirt = zip(m['params'],f['params'])
    # I then breed the numbers in the params columns using binary_mutant_sex. 
    new_params = [[binary_mutant_sex(tup,binary_rate) for tup in zip(sol[0],sol[1])] for 
                  sol in param_flirt]
    # Finally, I seperate out the children solutions into two new columns, one for each child chromosome.
    b_param, g_param = [[[sol[i] for sol in pair] for pair in new_params] for i in range(2)]
    
    # The next step is to breed the cutoff points using normal_mutant_sex. Again, I zip the columns of 
    # the parent chromosomes together.
    cutoff_flirt = zip(m['cutoffs'],f['cutoffs'])
    # Then I bread each pair.
    new_cutoffs = [normal_mutant_sex(tup,compressor,normal_rate) for tup in cutoff_flirt]
    # And seperate them into two columns.
    b_cutoff, g_cutoff = [[pair[i] for pair in new_cutoffs] for i in range(2)]
    
    # The last step is to breed the gls. I zip the gls columns of the parent chromosomes together.
    gl_flirt = list(zip(m['gls'],f['gls']))
    # Then I breed the gls by randomly selecting either one or the other gl, twice, to produce two
    # randomly mixed gl columns.
    b_gl,g_gl = [[np.random.choice([tup[0],tup[1]]) for tup in gl_flirt] for i in range(2)]
    
    # I put together the three pairs of columns just created together to produce two new chromosomes.
    return (pd.DataFrame([b_param,b_cutoff,b_gl],columns=m.index,index=['params','cutoffs','gls']).T,
            pd.DataFrame([g_param,g_cutoff,g_gl],columns=f.index,index=['params','cutoffs','gls']).T)
            
def dating_app(pop, ranked_index, choices):
    '''
    dating_app takes the population and the chosen ranked pairs and returns the selected parents. It also
    requires a ranked index that maps iloc to loc for parent rank.
    '''
    parent_index = [(ranked_index[tup[0]],ranked_index[tup[1]]) for tup in choices]
    return [(pop[tup[0]],pop[tup[1]]) for tup in parent_index]
    
def orgy(couple,compressor=4):
    '''
    Orgy is the complete sexual experience for a pair of solutions. It returns a pair of children
    solutions.
    '''
    # I assign thee pair to male and female
    m,f = couple[0], couple[1]
    # I use chromosex to create the new buy and sell chromosomes.
    buy_boy, buy_girl = chromosex(m[0],f[0],compressor)
    sell_boy, sell_girl = chromosex(m[1],f[1],compressor)
    # And directly apply binary_sex to get a new value for the number of stocks to invest in.
    stockno_boy, stockno_girl = binary_sex((m[2],f[2]))
    
    return ((buy_boy,sell_boy,stockno_boy),(buy_girl,sell_girl,stockno_girl))
    
def mutant_orgy(couple,compressor=4,binary_rate=0.01,normal_rate=0.015,gl_rate=0.01):
    '''
    Orgy is the complete mutant sexual experience for a pair of solutions. It returns a pair of 
    mutant children solutions. binary_rate is the mutation rate used in binary mutant sex 
    functions and normal rate is the rate used in normal mutant sex functions.
    '''
    # I assign thee pair to male and female
    m,f = couple[0], couple[1]
    # I use mutant_chromosex to create the new buy and sell chromosomes.
    buy_boy, buy_girl = mutant_chromosex(m[0],f[0],compressor,binary_rate,normal_rate,gl_rate)
    sell_boy, sell_girl = mutant_chromosex(m[1],f[1],binary_rate,normal_rate,gl_rate)
    # And directly apply binar_mutant_sex to get a new value for the number of stocks to invest in.
    stockno_boy, stockno_girl = binary_mutant_sex((m[2],f[2]),mutation_rate=binary_rate)
    
    return ((buy_boy,sell_boy,stockno_boy),(buy_girl,sell_girl,stockno_girl))

def validation_fitness(complete_solution,prices,closes,action_prices,ma_keys,bol_keys,growth_keys,
            supmax_keys,supmin_keys,return_keys,pvma_keys,delta_context,price_context,pv_context,
            min_density=0.5,spread=0.09/100,min_invest=50,start=5000,max_limit=5,proportion_limit=0.1):
    '''
    This fitness function takes a complete solution, which is a genome plus standard scaler objects and
    column indexes that contextualise the technicals dataframes, and returns the fitness for that complete
    solution. This fitness function must be set to have test data in order for it to return a useful
    result.
    '''
    
    # I split the complete_solution into two components: The genetic component, and the scalers and 
    # columns component. These are subsequently split into their constituent components.
    buy_chromosome, sell_chromosome, num_stocks = complete_solution[0]
    buy_scalers,buy_columns,sell_scalers,sell_columns = complete_solution[1]
    
    # I use the get_measures function to get the lists of technical dataframes, cutoffs, and gls.
    buy_technicals,buy_cutoffs,buy_gls,sell_technicals,sell_cutoffs,sell_gls = get_measures(
        buy_chromosome,sell_chromosome,prices,closes,ma_keys,bol_keys,growth_keys,supmax_keys,
        supmin_keys,return_keys,pvma_keys,delta_context,price_context,pv_context)
    
    # Supressing some useless warnings, I standardise the technicals dataframes using the standard
    # scalers contained in the complete solution. The columns are used to ensure that the standard
    # scaler objects are applied to the correct columns.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            buy_technicals = [pd.DataFrame(buy_scalers[i].transform(buy_technicals[i][buy_columns[i]]),
                                           index=buy_technicals[i].index,
                                           columns=buy_columns[i]) 
                                           for i in range(len(buy_scalers))] 
            sell_technicals = [pd.DataFrame(sell_scalers[i].transform(sell_technicals[i][sell_columns[i]]),
                                            index=sell_technicals[i].index,
                                            columns=sell_columns[i]) 
                                            for i in range(len(sell_scalers))]
        # I catch any value errors due to division by near zeros in technical dataframes.
        except ValueError:
            return np.nan
    # Using the resulting scaled technicals, I use OG_signal to produce a signals dataframe. I also
    # set the first valid index of the signals dataframes as first_signal.
    signals = OG_signal(buy_technicals,buy_cutoffs,buy_gls,sell_technicals,sell_cutoffs,sell_gls)
    first_signal = signals.first_valid_index()

    # If there are no buy signals, or there are too few signals relative to prices, or there are 
    # fewer than zero days between the first valid signal and the last available price date, then 
    # I return np.nan.
    if (len(signals[first_signal:]) / len(action_prices) < min_density or (signals==1).sum().sum() == 0 
        or (action_prices.last_valid_index() - first_signal).days <= 0):
        return np.nan
    # I then refine the signals and action_prices dataframes to include only relevant information.
    signals,action_prices = refiner(signals, action_prices, max_limit, proportion_limit)
    # Then I create the actions dictionary with action_maker...
    actions = action_maker(signals)
    # ...and run the simulation, returning its result.
    return simulation(actions, action_prices, start, num_stocks, min_invest, spread, first_signal)
    
# ------------------------------------------------------------- DIVINE FUNCTIONS -----------------------------------

def rand_bin_array(K, N):
    '''
    This function creates an array of length N with K ones randomly interspersed amongst zeros
    '''
    arr = np.zeros(N)
    arr[:K]  = 1
    np.random.shuffle(arr)
    return arr
    
def divine_starter(param_guide,max_ones,pantheon):
    '''
    The divine_starter function creates an initial population of buy and sell seed making arrays. It
    creates these arrays with a maximum of max_ones ones in each array, and creates a population of
    size equal to pantheon.
    '''
    metric_count = len(param_guide)
    return [(rand_bin_array(np.random.randint(1,max_ones),metric_count), 
             rand_bin_array(np.random.randint(1,max_ones),metric_count)) for i in range(pantheon)]
             
    
def zeus(couple,max_ones,metric_count):
    '''
    zeus, like its namesake, is the progenitor of the gods and is the sex function for the divinity
    level genetic algorithm. It breeds couples by randomly selecting between their binary values, and
    creates a random array with a maximum of max_ones ones in case the breeding process leads to an
    array with no ones.
    '''
    # I assign the inputted couple to male and female.
    m,f = couple[0],couple[1]
    # I breed the buy arrays, and then, the sell arrays, to produce a pair of children buy arrays
    # and a pair of children sell arays.
    buy_metrics = [[np.random.choice([tup[0],tup[1]]) for tup in zip(m[0],f[0])] for i in range(2)]
    sell_metrics = [[np.random.choice([tup[0],tup[1]]) for tup in zip(m[1],f[1])] for i in range(2)]
    # If any of these arrays have all zeros, I replace them with randomly generated arrays. 
    buy_metrics = [rand_bin_array(np.random.randint(1,max_ones),metric_count) if sum(sol) == 0 else sol 
                   for sol in buy_metrics]
    sell_metrics = [rand_bin_array(np.random.randint(1,max_ones),metric_count) if sum(sol) == 0 else sol 
                    for sol in sell_metrics]
    # I repackage the buy and sell pairs and return them as a couple of children solutions
    return ((buy_metrics[0],sell_metrics[0]),(buy_metrics[1],sell_metrics[1]))
    
def mutant_zeus(couple,max_ones,metric_count,mutation_rate=0.01):
    '''
    mutant_zeus is just like zeus except that it adds a mutation step that explores new solutions by
    randomly flipping bits at the mutation rate.
    '''
    # I assign the inputted couple to male and female.
    m,f = couple[0],couple[1]
    # I breed the buy arrays, and then, the sell arrays, to produce a pair of children buy arrays
    # and a pair of children sell arays.
    buy_metrics = [[np.random.choice([tup[0],tup[1]]) for tup in zip(m[0],f[0])] for i in range(2)]
    sell_metrics = [[np.random.choice([tup[0],tup[1]]) for tup in zip(m[1],f[1])] for i in range(2)]
    # for each array I generate a paired random array populated by ones at the mutation
    # rate. I flip the bits in the solution array if its paired array has a one.
    buy_mutations, sell_mutations = [[np.random.choice([1,0],p=[mutation_rate,1-mutation_rate],
                                                       size=metric_count) 
                                                       for i in range(2)] for i in range(2)]
    # Here is where I actually flip the bits.
    buy_metrics = [[1 - buy_metrics[i][k] if buy_mutations[i][k] == 1 else buy_metrics[i][k] for k in 
                    range(metric_count)] for i in range(2)]
    sell_metrics = [[1 - sell_metrics[i][k] if sell_mutations[i][k] == 1 else sell_metrics[i][k] for k in 
                     range(metric_count)] for i in range(2)]
    # If any of the resultant arrays have all zeros, I replace them with randomly generated arrays. 
    buy_metrics = [rand_bin_array(np.random.randint(1,max_ones),metric_count) if sum(sol) == 0 else sol 
                   for sol in buy_metrics]
    sell_metrics = [rand_bin_array(np.random.randint(1,max_ones),metric_count) if sum(sol) == 0 else sol 
                    for sol in sell_metrics]
    # I repackage the buy and sell pairs and return them as a couple of children solutions
    return ((buy_metrics[0],sell_metrics[0]),(buy_metrics[1],sell_metrics[1]))
    
def uniquify(pop):
    '''
    This function ensures a divine population only has unique members.
    '''
    pop = [(list(tup[0]),list(tup[1])) for tup in pop]
    unique_pop = []
    for sol in pop:
        if sol not in unique_pop:
            unique_pop.append(sol)
    return unique_pop
    
