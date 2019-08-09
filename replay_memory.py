import h5py
import math
import numpy as np
import pdb
import random
import progressbar

# Class to load and preprocess data
class ReplayMemory():
    def __init__(self, args, shift, scale, shift_u, scale_u, env, net, sess):
        self.batch_size = args.batch_size
        self.seq_length = 2*args.seq_length
        self.shift_x = shift
        self.scale_x = scale
        self.shift_u = shift_u
        self.scale_u = scale_u
        self.env = env
        self.net = net
        self.sess = sess

        print('validation fraction: ', args.val_frac)

        print("generating data...")
        self._generate_data(args)
        self._create_inputs_targets(args)

        print('creating splits...')
        self._create_split(args)

        print('shifting/scaling data...')
        self._shift_scale(args)

    def _generate_data(self, args):
        # Initialize array to hold states and actions
        x = np.zeros((args.n_trials, args.n_subseq, self.seq_length, args.state_dim), dtype=np.float32)
        u = np.zeros((args.n_trials, args.n_subseq, self.seq_length-1, args.action_dim), dtype=np.float32)

        # Define progress bar
        bar = progressbar.ProgressBar(maxval=args.n_trials).start()

        # Define array for dividing trials into subsequences
        stagger = (args.trial_len - self.seq_length)/args.n_subseq
        self.start_idxs = np.linspace(0, stagger*args.n_subseq, args.n_subseq)

        # Loop through episodes
        for i in range(args.n_trials):
            # Define arrays to hold observed states and actions in each trial
            x_trial = np.zeros((args.trial_len, args.state_dim), dtype=np.float32)
            u_trial = np.zeros((args.trial_len-1, args.action_dim), dtype=np.float32)

            # Reset environment and simulate with random actions
            x_trial[0] = self.env.reset()
            for t in range(1, args.trial_len): 
                action = self.env.action_space.sample()  
                u_trial[t-1] = action
                step_info = self.env.step(action)
                x_trial[t] = np.squeeze(step_info[0])

            # Divide into subsequences
            for j in range(args.n_subseq):
                x[i, j] = x_trial[int(self.start_idxs[j]):(int(self.start_idxs[j])+self.seq_length)]
                u[i, j] = u_trial[int(self.start_idxs[j]):(int(self.start_idxs[j])+self.seq_length-1)]
            bar.update(i)
        bar.finish()


        # Generate test scenario that is double the length of standard sequences
        self.x_test = np.zeros((2*args.seq_length, args.state_dim), dtype=np.float32)
        self.u_test = np.zeros((2*args.seq_length-1, args.action_dim), dtype=np.float32)
        self.x_test[0] = self.env.reset()
        for t in range(1, 2*args.seq_length):
            action = self.env.action_space.sample()
            self.u_test[t-1] = action
            step_info = self.env.step(action)
            self.x_test[t] = np.squeeze(step_info[0])

        # Reshape and trim data sets
        self.x = x.reshape(-1, self.seq_length, args.state_dim)
        self.u = u.reshape(-1, self.seq_length-1, args.action_dim)
        len_x = int(np.floor(len(self.x)/args.batch_size)*args.batch_size)
        self.x = self.x[:len_x]
        self.u = self.u[:len_x]

    def _create_inputs_targets(self, args):
        # Create batch_dict
        self.batch_dict = {}

        # Print tensor shapes
        print('states: ', self.x.shape)
        print('inputs: ', self.u.shape)
            
        self.batch_dict['states'] = np.zeros((args.batch_size, self.seq_length, args.state_dim))
        self.batch_dict['inputs'] = np.zeros((args.batch_size, self.seq_length-1, args.action_dim))

        # Shuffle data before splitting into train/val
        print('shuffling...')
        p = np.random.permutation(len(self.x))
        self.x = self.x[p]
        self.u = self.u[p]

    # Separate data into train/validation sets
    def _create_split(self, args):
        # Compute number of batches
        self.n_batches = len(self.x)//args.batch_size
        self.n_batches_val = int(math.floor(args.val_frac * self.n_batches))
        self.n_batches_train = self.n_batches - self.n_batches_val

        print('num training batches: ', self.n_batches_train)
        print('num validation batches: ', self.n_batches_val)

        # Divide into train and validation datasets
        self.x_val = self.x[self.n_batches_train*args.batch_size:]
        self.u_val = self.u[self.n_batches_train*args.batch_size:]
        self.x = self.x[:self.n_batches_train*args.batch_size]
        self.u = self.u[:self.n_batches_train*args.batch_size]

        # Set batch pointer for training and validation sets
        self.reset_batchptr_train()
        self.reset_batchptr_val()

    # Shift and scale data to be zero-mean, unit variance
    def _shift_scale(self, args):
        # Find means and std if not initialized to anything
        if np.sum(self.scale_x) == 0.0:
            self.shift_x = np.mean(self.x[:self.n_batches_train], axis=(0, 1))
            self.scale_x = np.std(self.x[:self.n_batches_train], axis=(0, 1))
            self.shift_u = np.mean(self.u[:self.n_batches_train], axis=(0, 1))
            self.scale_u = np.std(self.u[:self.n_batches_train], axis=(0, 1))

            # Remove very small scale values
            self.scale_x[self.scale_x < 1e-6] = 1.0

        # Shift and scale values for test sequence
        self.x_test = (self.x_test - self.shift_x)/self.scale_x
        self.u_test = (self.u_test - self.shift_u)/self.scale_u

    # Sample a new batch of data
    def next_batch_train(self):
        # Extract next batch
        batch_index = self.batch_permuation_train[self.batchptr_train*self.batch_size:(self.batchptr_train+1)*self.batch_size]
        self.batch_dict['states'] = (self.x[batch_index] - self.shift_x)/self.scale_x
        self.batch_dict['inputs'] = (self.u[batch_index] - self.shift_u)/self.scale_u

        # Update pointer
        self.batchptr_train += 1
        return self.batch_dict

    # Return to first batch in train set
    def reset_batchptr_train(self):
        self.batch_permuation_train = np.random.permutation(len(self.x))
        self.batchptr_train = 0

    # Return next batch of data in validation set
    def next_batch_val(self):
        # Extract next validation batch
        batch_index = range(self.batchptr_val*self.batch_size,(self.batchptr_val+1)*self.batch_size)
        self.batch_dict['states'] = (self.x_val[batch_index] - self.shift_x)/self.scale_x
        self.batch_dict['inputs'] = (self.u_val[batch_index] - self.shift_u)/self.scale_u

        # Update pointer
        self.batchptr_val += 1
        return self.batch_dict

    # Return to first batch in validation set
    def reset_batchptr_val(self):
        self.batchptr_val = 0

