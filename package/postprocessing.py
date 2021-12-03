import matplotlib.pyplot as plt
from numpy.lib.shape_base import column_stack
from numpy import empty, mean
from pandas import DataFrame
from preprocessing import pairs, insts
from random import randrange

def mape(actual, forecast):
    '''Mean Absolute Percentage Error'''
    return mean(abs((forecast - actual) / actual)) * 100

def divide_currencies(predictions):
    '''predict currency pair by dividing the predictions of its consituent
    currencies.'''
    divided = empty((predictions.shape[0], 28))
    for i, pair in enumerate(pairs):
        base, quote = pair[:3], pair[3:]
        b, q = insts.index(base), insts.index(quote)
        divided[:, i] = (predictions[:, b] / predictions[:, q]).flatten()
    return divided

def print_results(ct, pt, currency_accuracy, pair_accuracy, mape_improvement):
    sumc = sum(ct)
    sump = sum(pt)
    print(mape_improvement)
    print(f'Improved results for {mape_improvement[mape_improvement.improvement > 0].count()[0]}/28 currency pairs')
    print('Mean Absolute Percentage Errors:')
    print(f'Pair: {round(mean(pair_accuracy), 2)}%')
    print(f'Currency: {round(mean(currency_accuracy), 2)}% (Lower is better)')
    print(f'Took {round(sump)} sec to process pairs but only \
    {round(sumc)} sec to process currencies...')
    print("... shaving off {:.0%}".format((sump-sumc) / sump))

def plot_random(y, predictions, divided=None, pair=-1):    
    columns = ['Actual', 'Direct prediction']
    if divided is None:
        cstack = [y, predictions]
    else:
        if pair < 0: pair = randrange(28)
        cstack = [y.iloc[:, 8+pair], predictions[:, 8+pair], divided[:, pair]]
        columns.append('Divided currencies')
    d = DataFrame(column_stack(cstack), index=y.index, columns=columns)
    plt.plot(d, label=columns)
    plt.title(pairs[pair])
    plt.legend()
    plt.show()
