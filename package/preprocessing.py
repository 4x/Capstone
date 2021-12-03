import pickle
from pandas import DataFrame, merge

# Global lists
insts = ["AUD", "NZD", "EUR", "GBP", "CAD", "CHF", "JPY", "USD"]
pairs = ['AUDNZD', 'EURAUD', 'GBPAUD', 'AUDCAD', 'AUDCHF', 'AUDJPY', 'AUDUSD',\
     'EURNZD', 'GBPNZD', 'NZDCAD', 'NZDCHF', 'NZDJPY', 'NZDUSD', 'EURGBP',\
        'EURCAD', 'EURCHF', 'EURJPY', 'EURUSD', 'GBPCAD', 'GBPCHF', 'GBPJPY',\
        'GBPUSD', 'CADCHF', 'CADJPY', 'USDCAD', 'CHFJPY', 'USDCHF', 'USDJPY']
#n_insts = len(insts)
pair_map = [[0, 3, 4, 5, 6], [9, 10, 11, 12], [1, 7, 13, 14, 15, 16, 17],\
    [2, 8, 18, 19, 20, 21], [22, 23], [25], [], [24, 26, 27]]

def map_pairs_to_currency(insts, pairs):
    matching_pairs = list()
    #for i, c1 in enumerate(insts):
    for c1 in insts:
        p = list()
        for j, pair in enumerate(pairs):
            if pair[:3] == c1:
                p.append(j)
        matching_pairs.append(p)
    assert len(matching_pairs) == len(insts)
    print(matching_pairs)
    return matching_pairs

def create_inclusive_array(freq='30Min', year=2019, features=1, C=True,P=True):
    '''Put all currencies and/or all pairs into one dataframe, so that the time
    index exactly aligns: this makes comparisons easier.
    4 features → OHLC
    3 features → HLC
    1 feature → C only.'''
    df = DataFrame()
    col = 0
    if C: # include currencies
        for c1 in insts:
            print(c1)
            with open("./"+freq+"_"+str(year)+"/"+c1 +'_'+freq+'.pickle','rb')\
            as pickle_file:
                d = pickle.load(pickle_file).iloc[:,-features:]
            df = merge(df,d, left_index=True,right_index=True, how='outer',\
                    suffixes=(None, c1))
        col += len(insts) * features # 4 columns per currency
    if P: # include pairs
        for c1 in pairs:
            print(c1)
            with open("./"+freq+"_"+str(year)+"/"+c1 +'_'+freq+'.pickle','rb')\
            as pickle_file:
                d = pickle.load(pickle_file).iloc[:,-features:]
            df = merge(df,d, left_index=True, right_index=True,how='outer',\
                    suffixes=(None, c1))
        col += len(pairs) * features # 4 columns per currency
    assert df.shape[1] == col
    return df.fillna(method="ffill").fillna(method="bfill").dropna()
