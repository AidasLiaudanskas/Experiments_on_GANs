import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import collections
import time
import pickle as pickle

_since_beginning = collections.defaultdict(lambda: {})
_since_last_flush = collections.defaultdict(lambda: {})

_iter = [0]
def tick():
	_iter[0] += 1

def plot(name, value):
	_since_last_flush[name][_iter[0]] = value

def flush():
	prints = []

	for name, vals in list(_since_last_flush.items()):
		prints.append("{} = {}".format(name, np.mean(list(vals.values()))))
		_since_beginning[name].update(vals)

		x_vals = np.sort(list(_since_beginning[name].keys()))
		y_vals = [_since_beginning[name][x] for x in x_vals]

		plt.clf()
		plt.plot(x_vals, y_vals)
		plt.xlabel('iteration')
		plt.ylabel(name)
		plt.savefig(name.replace(' ', '_')+'.jpg')

	print(("Step:{:6}|  {:9}".format(_iter[0], " | ".join(prints))))
	                        # print('Model: {:15} Global step: {:9}   Step number {:8} out of {:8} in the current session.  '
	                        #       'Time taken {:6.4f}  Accuracy: {:6.6f}  '
	                        #       'Loss: {:6.6f}'.format(model, step, step - init_step, max_steps,
	                        #                              ((time.time() - start_time)), acc,
	                        #                              loss))


	_since_last_flush.clear()

	with open('log.pkl', 'wb') as f:
		pickle.dump(dict(_since_beginning), f, pickle.HIGHEST_PROTOCOL)
