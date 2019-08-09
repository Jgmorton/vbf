import h5py
import math
import matplotlib.pyplot as plt
import numpy as np
import os, sys
import progressbar
import pdb
import tensorflow as tf

# Plot predictions against true time evolution
def visualize_predictions(args, sess, net, replay_memory, env, e=0):
	# Get inputs (test trajectory that is twice the size of a standard sequence)
    x = np.zeros((args.batch_size, 2*args.seq_length, args.state_dim), dtype=np.float32)
    u = np.zeros((args.batch_size, 2*args.seq_length-1, args.action_dim), dtype=np.float32)
    x[:] = replay_memory.x_test
    u[:] = replay_memory.u_test

    # Find number of times to feed in input
    n_passes = 200//args.batch_size

    # Initialize array to hold predictions
    preds = np.zeros((1, 2*args.seq_length, args.state_dim))
    for t in range(n_passes):
        # Construct inputs for network
        feed_in = {}
        feed_in[net.x] = np.reshape(x, (2*args.batch_size*args.seq_length, args.state_dim))
        feed_in[net.u] = u
        feed_out = net.state_pred
        out = sess.run(feed_out, feed_in)
        x_pred = out.reshape(args.batch_size, 2*args.seq_length, args.state_dim)

        # Append new set of predictions
        preds = np.concatenate((preds, x_pred), axis=0)  
    preds = preds[1:]

    # Find mean, max, and min of predictions
    pred_mean = np.mean(preds, axis=0)
    pred_std = np.std(preds, axis=0)
    pred_min = np.amin(preds, axis=0)
    pred_max = np.amax(preds, axis=0)

    diffs = np.linalg.norm((preds[:, :args.seq_length] - sess.run(net.shift))/sess.run(net.scale) - x[0, :args.seq_length], axis=(1, 2))
    best_pred = np.argmin(diffs)
    worst_pred = np.argmax(diffs)
        
    # Plot different quantities
    x = x*sess.run(net.scale) + sess.run(net.shift)

    # Find indices for random predicted trajectories to plot
    ind0 = best_pred
    ind1 = worst_pred

    # Plot values
    plt.close()
    f, axs = plt.subplots(args.state_dim, sharex=True, figsize=(15, 15))
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    for i in range(args.state_dim):
        axs[i].plot(range(2*args.seq_length), x[0, :, i], 'k')
        axs[i].plot(range(2*args.seq_length), preds[ind0, :, i], 'r')
        axs[i].plot(range(2*args.seq_length), preds[ind1, :, i], 'g')
        axs[i].plot(range(2*args.seq_length), pred_mean[:, i], 'b')
        axs[i].fill_between(range(2*args.seq_length), pred_min[:, i], pred_max[:, i], facecolor='blue', alpha=0.5)
        axs[i].set_ylim([np.amin(x[0, :, i])-0.2, np.amax(x[0, :, i]) + 0.2])

    plt.xlabel('Time Step')
    plt.xlim([0, 2*args.seq_length])
    plt.savefig('bf_predictions/predictions_' + str(e) + '.png')
