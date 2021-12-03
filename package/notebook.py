import functions
import preprocessing
import postprocessing

# Find optimal hyperparameters
units, learning_rate, best_epoch = prepare_run_hypersearch(35)

# Create one dataframe with all currencies and pairs to line up date/times
df = preprocessing.create_inclusive_array(freq='30Min', year=2019)
predictions, divided, y, mapes, mape_improvement = functions.envelope(df)
y, predictions = functions.envelope(df)
y[:, 0], predictions[:, 0] = functions.envelope(df)

prepare_run_hypersearch(df)

plot_random(y, predictions, divided, pair=1)
postprocessing.plot_random(y, predictions)

# train/predict only one pair
pair_map = list()
pair_map.append([])
y, predictions = envelope(df.iloc[:, 0:1])
plot_random(y, predictions)