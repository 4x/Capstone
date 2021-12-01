


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


currency_accuracy, currency_frcast, currency_actual, ctime,\
        pair_accuracy, pair_frcast, pair_actual, ptime =\
    distribute_predictions(prefix='./30Min_2019/', suffix = '_30Min.pickle')

df = create_inclusive_array()
predictions, unscaled_y, ctime, ptime, mapes = distribute_predictions(prefix='./30Min_2019/', suffix = '_30Min.pickle')

df = create_inclusive_array()
predictions, unscaled_y, ctime, ptime, mapes = distribute_predictions(df)

units, learning_rate, best_epoch = prepare_run_hypersearch(35)

divided = divide_currencies(predictions)
predicted_pairs = predictions[:, 8:]
true_pairs = np.array(unscaled_y.iloc[:, 8:])
mape(true_pairs, divided)
mape(true_pairs, predicted_pairs)

d = true_pairs - predicted_pairs
d = d / true_pairs
m = abs(d)
m = mean(d.flatten())

