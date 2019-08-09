import numpy as np
import tensorflow as tf

class BayesFilter():
    def __init__(self, args):

        # Placeholder for data -- inputs are number of elements x pts in mesh x dimensionality of data for each point
        self.x = tf.Variable(np.zeros((2*args.batch_size*args.seq_length, args.state_dim), dtype=np.float32), trainable=False, name="state_values")
        self.u = tf.Variable(np.zeros((args.batch_size, 2*args.seq_length-1, args.action_dim), dtype=np.float32), trainable=False, name="action_values")

        # Parameters to be set externally
        self.learning_rate = tf.Variable(0.0, trainable=False, name="learning_rate")
        self.kl_weight = tf.Variable(0.0, trainable=False, name="kl_weight")

        # Normalization parameters to be stored
        self.shift = tf.Variable(np.zeros(args.state_dim), trainable=False, name="input_shift", dtype=tf.float32)
        self.scale = tf.Variable(np.zeros(args.state_dim), trainable=False, name="input_scale", dtype=tf.float32)
        self.shift_u = tf.Variable(np.zeros(args.action_dim), trainable=False, name="action_shift", dtype=tf.float32)
        self.scale_u = tf.Variable(np.zeros(args.action_dim), trainable=False, name="action_scale", dtype=tf.float32)
        self.generative = tf.Variable(False, trainable=False, name="generate_flag")
        
        # Create the computational graph
        self._create_feature_extractor_params(args)
        self._create_feature_extractor(args)
        self._create_initial_generator(args)
        self._create_transition_matrices(args)
        self._create_weight_network_params(args)
        self._create_inference_network_params(args)
        self._propagate_solution(args)
        self._create_decoder_params(args)
        self._create_optimizer(args)

    # Create parameters to comprise inference network
    def _create_feature_extractor_params(self, args):
        self.extractor_w = []
        self.extractor_b = []

        # Loop through elements of decoder network and define parameters
        for i in range(len(args.extractor_size)):
            if i == 0:
                prev_size = args.state_dim
            else:
                prev_size = args.extractor_size[i-1]
            self.extractor_w.append(tf.get_variable("extractor_w"+str(i), [prev_size, args.extractor_size[i]], 
                                                regularizer=tf.contrib.layers.l2_regularizer(args.reg_weight)))
            self.extractor_b.append(tf.get_variable("extractor_b"+str(i), [args.extractor_size[i]]))

        # Last set of weights to map to output
        self.extractor_w.append(tf.get_variable("extractor_w_end", [args.extractor_size[-1], args.code_dim], 
                                                regularizer=tf.contrib.layers.l2_regularizer(args.reg_weight)))
        self.extractor_b.append(tf.get_variable("extractor_b_end", [args.code_dim]))

    # Function to run inputs through extractor
    def _get_extractor_output(self, args, states):
        extractor_input = states
        for i in range(len(args.extractor_size)):
            extractor_input = tf.nn.relu(tf.nn.xw_plus_b(extractor_input, self.extractor_w[i], self.extractor_b[i]))
        output = tf.nn.xw_plus_b(extractor_input, self.extractor_w[-1], self.extractor_b[-1])
        return output

    # Create feature extractor (maps state -> features, assumes feature same dimensionality as latent states)
    def _create_feature_extractor(self, args):
        features = self._get_extractor_output(args, self.x)
        self.features = tf.reshape(features, [args.batch_size, 2*args.seq_length, args.code_dim])

    # Function to generate samples given distribution parameters
    def _gen_sample(self, args, dist_params):
        w_mean, w_logstd = tf.split(dist_params, [args.noise_dim, args.noise_dim], axis=1)
        w_std = tf.exp(w_logstd) + 1e-3
        samples = tf.random_normal([args.batch_size, args.noise_dim])
        w = samples*w_std + w_mean
        w = tf.minimum(tf.maximum(w, -10.0), 10.0)
        w = tf.cond(self.generative, lambda: samples, lambda: w)  # Just sample from prior for generative model
        return w

    # Bidirectional LSTM to generate initial sample of w1, then form z1 from w1
    def _create_initial_generator(self, args):
        fwd_cell = tf.nn.rnn_cell.LSTMCell(args.rnn_size, initializer=tf.contrib.layers.xavier_initializer())
        bwd_cell = tf.nn.rnn_cell.LSTMCell(args.rnn_size, initializer=tf.contrib.layers.xavier_initializer())
        
        # Get outputs from rnn and concatenate
        outputs, _ = tf.nn.bidirectional_dynamic_rnn(fwd_cell, bwd_cell, self.features[:, :args.seq_length], dtype=tf.float32)
        output_fw, output_bw = outputs
        output = tf.concat([output_fw[:, -1], output_bw[:, -1]], axis=1)

        # Single affine transformation into w1 distribution params
        hidden = tf.layers.dense(output, 
                                args.transform_size, 
                                activation=tf.nn.relu,
                                name='to_hidden_w1', 
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(args.reg_weight))
        self.w1_dist = tf.layers.dense(hidden, 
                                2*args.noise_dim, 
                                name='to_w1', 
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(args.reg_weight))
        self.w1 = self._gen_sample(args, self.w1_dist)

        # Now construct z1 through transformation with single hidden layer
        hidden = tf.layers.dense(self.w1, 
                                args.transform_size, 
                                activation=tf.nn.relu,
                                name='to_hidden_z1', 
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(args.reg_weight))
        self.z1 = tf.layers.dense(hidden, 
                                args.code_dim, 
                                name='to_z1', 
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(args.reg_weight))

    # Initialize potential transition matrices
    def _create_transition_matrices(self, args):
        self.A_matrices = tf.get_variable("A_matrices", [args.num_matrices, args.code_dim, args.code_dim], 
                                            regularizer=tf.contrib.layers.l2_regularizer(args.reg_weight))
        self.B_matrices = tf.get_variable("B_matrices", [args.num_matrices, args.action_dim, args.code_dim], 
                                            regularizer=tf.contrib.layers.l2_regularizer(args.reg_weight))
        self.C_matrices = tf.get_variable("C_matrices", [args.num_matrices, args.noise_dim, args.code_dim], 
                                            regularizer=tf.contrib.layers.l2_regularizer(args.reg_weight))

    # Create parameters to comprise weight network
    def _create_weight_network_params(self, args):
        self.weight_w = []
        self.weight_b = []

        # Have single hidden layer and fully connected to output
        self.weight_w.append(tf.get_variable("weight_w1", [args.code_dim+args.action_dim, args.transform_size], 
                                            regularizer=tf.contrib.layers.l2_regularizer(args.reg_weight)))
        self.weight_w.append(tf.get_variable("weight_w2", [args.transform_size, args.num_matrices], 
                                            regularizer=tf.contrib.layers.l2_regularizer(args.reg_weight)))

        self.weight_b.append(tf.get_variable("weight_b1", [args.transform_size], 
                                            regularizer=tf.contrib.layers.l2_regularizer(args.reg_weight)))
        self.weight_b.append(tf.get_variable("weight_b2", [args.num_matrices], 
                                            regularizer=tf.contrib.layers.l2_regularizer(args.reg_weight)))

    # Create parameters to comprise inference network
    def _create_inference_network_params(self, args):
        self.inference_w = []
        self.inference_b = []

        # Loop through elements of inference network and define parameters
        for i in range(len(args.inference_size)):
            if i == 0:
                prev_size = args.feature_dim+args.code_dim+args.action_dim
            else:
                prev_size = args.inference_size[i-1]
            self.inference_w.append(tf.get_variable("inference_w"+str(i), [prev_size, args.inference_size[i]], 
                                                regularizer=tf.contrib.layers.l2_regularizer(args.reg_weight)))
            self.inference_b.append(tf.get_variable("inference_b"+str(i), [args.inference_size[i]]))

        # Last set of weights to map to output
        self.inference_w.append(tf.get_variable("inference_w_end", [args.inference_size[-1], 2*args.noise_dim], 
                                                regularizer=tf.contrib.layers.l2_regularizer(args.reg_weight)))
        self.inference_b.append(tf.get_variable("inference_b_end", [2*args.noise_dim]))

    # Function to get weights for transition matrices
    def _get_weights(self, z, u):
        z_u = tf.concat([z, u], axis=1)
        hidden = tf.nn.relu(tf.nn.xw_plus_b(z_u, self.weight_w[0], self.weight_b[0]))
        return tf.nn.softmax(tf.nn.xw_plus_b(hidden, self.weight_w[1], self.weight_b[1]))

    # Function to generate w sample from inference network
    def _get_inference_sample(self, args, features, z, u):
        inference_input = tf.concat([features, z, u], axis=1)
        for i in range(len(args.inference_size)):
            inference_input = tf.nn.relu(tf.nn.xw_plus_b(inference_input, self.inference_w[i], self.inference_b[i]))
        w_dist = tf.nn.xw_plus_b(inference_input, self.inference_w[-1], self.inference_b[-1])

        # Generate sample
        w = self._gen_sample(args, w_dist)
        return w_dist, w

    # Now use various params/networks to propagate solution forward in time
    def _propagate_solution(self, args):
        # Find current observation (expand dimension for stacking later)
        z_t = tf.expand_dims(self.z1, axis=1)

        # Initialize array for stacking observations and distribution params
        self.z_pred = [z_t]
        self.w_dists = [tf.expand_dims(self.w1_dist, axis=1)]

        # Loop through time and advance observation, get distribution params
        for t in range(1, args.seq_length):
            # Get action
            u_t = self.u[:, t-1]

            # Find A, B, and C matrices
            weights = self._get_weights(tf.squeeze(z_t, axis=1), u_t)
            A_t_list = []
            B_t_list = []
            C_t_list = []
            for i in range(args.batch_size):
                A_t_list.append(tf.add_n([weights[i, j]*self.A_matrices[j] for j in range(args.num_matrices)])) 
                B_t_list.append(tf.add_n([weights[i, j]*self.B_matrices[j] for j in range(args.num_matrices)])) 
                C_t_list.append(tf.add_n([weights[i, j]*self.C_matrices[j] for j in range(args.num_matrices)]))
            A_t = tf.stack(A_t_list) 
            B_t = tf.stack(B_t_list)
            C_t = tf.stack(C_t_list) 

            # Draw noise sample and append sample to list
            w_dist, w_t = self._get_inference_sample(args, self.features[:, t], tf.squeeze(z_t, axis=1), u_t)
            self.w_dists.append(tf.expand_dims(w_dist, axis=1))

            # Now advance observation forward in time
            u_t = tf.expand_dims(u_t, axis=1)
            z_t = tf.matmul(z_t, A_t) + tf.matmul(u_t, B_t) + tf.matmul(tf.expand_dims(w_t, axis=1), C_t)
            self.z_pred.append(z_t)

        # Remove this part to have alg from paper
        for t in range(args.seq_length, 2*args.seq_length):
            # Find A, B, and C matrices
            u_t = self.u[:, t-1]
            weights = self._get_weights(tf.squeeze(z_t, axis=1), u_t)
            A_t_list = []
            B_t_list = []
            C_t_list = []
            for i in range(args.batch_size):
                A_t_list.append(tf.add_n([weights[i, j]*self.A_matrices[j] for j in range(args.num_matrices)])) 
                B_t_list.append(tf.add_n([weights[i, j]*self.B_matrices[j] for j in range(args.num_matrices)])) 
                C_t_list.append(tf.add_n([weights[i, j]*self.C_matrices[j] for j in range(args.num_matrices)]))
            A_t = tf.stack(A_t_list) 
            B_t = tf.stack(B_t_list)
            C_t = tf.stack(C_t_list) 

            # Draw noise sample and append sample to list
            w_t = tf.random_normal([args.batch_size, args.noise_dim])

            # Now advance observation forward in time
            u_t = tf.expand_dims(u_t, axis=1)
            z_t = tf.matmul(z_t, A_t) + tf.matmul(u_t, B_t) + tf.matmul(tf.expand_dims(w_t, axis=1), C_t)
            self.z_pred.append(z_t)

        # Stack predictions into single tensor for reconstruction
        self.z_pred = tf.reshape(tf.stack(self.z_pred, axis=1), [2*args.batch_size*args.seq_length, args.code_dim])
        self.w_dists = tf.reshape(tf.stack(self.w_dists, axis=1), [args.batch_size*args.seq_length, 2*args.noise_dim])
        
    # Create parameters to comprise decoder network
    def _create_decoder_params(self, args):
        self.decoder_w = []
        self.decoder_b = []

        # Loop through elements of decoder network and define parameters
        for i in range(len(args.extractor_size)-1, -1, -1):
            if i == len(args.extractor_size)-1:
                prev_size = args.code_dim
            else:
                prev_size = args.extractor_size[i+1]
            self.decoder_w.append(tf.get_variable("decoder_w"+str(len(args.extractor_size)-i), [prev_size, args.extractor_size[i]], 
                                                regularizer=tf.contrib.layers.l2_regularizer(args.reg_weight)))
            self.decoder_b.append(tf.get_variable("decoder_b"+str(len(args.extractor_size)-i), [args.extractor_size[i]]))

        # Last set of weights to map to output
        self.decoder_w.append(tf.get_variable("decoder_w_end", [args.extractor_size[0], args.state_dim], 
                                                regularizer=tf.contrib.layers.l2_regularizer(args.reg_weight)))
        self.decoder_b.append(tf.get_variable("decoder_b_end", [args.state_dim]))

    # Function to run inputs through decoder
    def _get_decoder_output(self, args, encodings):
        decoder_input = encodings
        for i in range(len(args.extractor_size)):
            decoder_input = tf.nn.relu(tf.nn.xw_plus_b(decoder_input, self.decoder_w[i], self.decoder_b[i]))
        output = tf.nn.xw_plus_b(decoder_input, self.decoder_w[-1], self.decoder_b[-1])
        return output

    # Create optimizer to minimize loss
    def _create_optimizer(self, args):
        # Find reconstruction loss
        self.rec_sol = self._get_decoder_output(args, self.z_pred)
        self.loss_reconstruction = tf.reduce_sum(tf.square(self.x - self.rec_sol))

        # Find state predictions by undoing data normalization
        self.state_pred = self.rec_sol*self.scale + self.shift

        # Find KL-divergence component of loss
        w_mean, w_logstd = tf.split(self.w_dists, [args.noise_dim, args.noise_dim], axis=1)
        w_std = tf.exp(w_logstd) + 1e-3

        # Define distribution and prior objects
        w_dist = tf.distributions.Normal(loc=w_mean, scale=w_std)
        prior_dist = tf.distributions.Normal(loc=tf.zeros_like(w_mean), scale=tf.ones_like(w_std))
        self.kl_loss = tf.reduce_sum(tf.distributions.kl_divergence(w_dist, prior_dist))

        # Sum with regularization losses to form total cost
        self.cost = self.loss_reconstruction + tf.reduce_sum(tf.losses.get_regularization_losses()) + self.kl_weight*self.kl_loss

        # Perform parameter update
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        tvars = tf.trainable_variables()
        self.grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), args.grad_clip)
        self.train = optimizer.apply_gradients(zip(self.grads, tvars))




