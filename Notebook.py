

# Find optimal hyperparameters
units, learning_rate, best_epoch = prepare_run_hypersearch(35)

# Create one dataframe with all currencies and pairs to line up date/times
df = create_inclusive_array(freq='30Min', year=2019)
predictions, divided, unscaled_y, mapes, mape_improvement = envelope(df)

prepare_run_hypersearch(df)

plot_random(unscaled_y, predictions, divided, pair=8)
