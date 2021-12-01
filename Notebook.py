
# Create one dataframe with all currencies and pairs to line up date/times
df = create_inclusive_array(freq='30Min', year=2019)

# Find optimal hyperparameters
units, learning_rate, best_epoch = prepare_run_hypersearch(35)

predictions, unscaled_y, ctime, ptime, mapes, compa = envelope(df)


