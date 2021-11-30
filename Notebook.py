


# Verify
with open("GBP_" + freq + ".pickle", 'rb') as pickle_file:
    base = pickle.load(pickle_file)

# Save 2019 M30 results
with open("2019_M30currency_frcst", "wb") as pickle_file:
    pickle.dump(currency_frcast, pickle_file)
with open("2019_M30pair_frcst", "wb") as pickle_file:
    pickle.dump(pair_frcast, pickle_file)

df = create_inclusive_array(freq='30Min', year=2019)

f = ".\30Min_2019\AUD_30Min.pickle"
freq = '30Min'
year = 2019
f = './' + freq + '_' + str(year) + '/' + insts[1] + '_' + freq + '.pickle'
with open(f, "rb") as pickle_file:
    df = pickle.load(pickle_file)

pipeline12(f)

syn_forecast, syn_accuracy, stime = syn_forecasts()

for i in currency_frcast:
    print(i.shape)
for i in pair_actual:
    print(i.shape)

df = create_inclusive_array()

currency_pairs = create_currency_pair_list()
currency_accuracy, currency_frcast, currency_actual, ctime,\
        pair_accuracy, pair_frcast, pair_actual, ptime =\
    distribute_predictions(prefix='./30Min_2019/', suffix = '_30Min.pickle')
sitp = distribute_predictions(prefix='./30Min_2019/', suffix = '_30Min.pickle')
