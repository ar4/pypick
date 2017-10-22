'''Predict picks.'''
import numpy as np
from scipy.interpolate import interp1d
import tensorflow as tf
from sklearn.linear_model import LinearRegression
from sklearn.base import BaseEstimator, RegressorMixin

class Fine_reg_tensorflow(BaseEstimator, RegressorMixin):
    '''A regression estimator object with fit and predict methods to predict
    picks given approximate picks using a CNN.
    '''
    def __init__(self, box_size=15, layer_len=5, num_steps=1000, batch_size=1,
                 interp_factor=100, print_interval=None):
        '''Arguments:
          box_size: integer specifying length of windowed extracted around
            approximate picks
          layer_len: integer specifying length of CNN layer. Should probably be
            smaller than box_size
          num_steps: integer specifying number of training steps to run
          batch_size: integer specifying size of batches supplied to training
          interp_factor: integer specifying factor to use when interpolating
            the data to allow non-integer picks. If set to 10, the finest
            interval between possible picks is 0.1, for example.
          print_interval: integer specifying number of number of training
            steps between printing of loss. If None, never print.
        '''
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.interp_factor = interp_factor
        self._box_size = box_size
        self._layer_len = layer_len
        self.box_size = box_size
        self.layer_len = layer_len
        self.print_interval = print_interval

    def _tf_setup(self):
        tf.reset_default_graph()
        self.sess = tf.Session()
        self.box = tf.placeholder(tf.float32, [None, self.box_size, 1])
        self.y_true = tf.placeholder(tf.float32, [None, self.box_size])
        self.layer1 = tf.layers.conv1d(self.box, 1, self.layer_len,
                                       padding='SAME',
                                       name='layer1')
        self.layer1 = tf.squeeze(self.layer1, 2)
        self.loss = tf.losses.mean_squared_error(labels=self.y_true,
                                                 predictions=self.layer1)
        self.optimizer = tf.train.AdamOptimizer().minimize(self.loss)

    @property
    def box_size(self):
        return self._box_size

    @box_size.setter
    def box_size(self, box_size):
        self._box_size = box_size
        self.num_interp = self.interp_factor * self.box_size
        self._tf_setup()

    @property
    def layer_len(self):
        return self._layer_len

    @layer_len.setter
    def layer_len(self, layer_len):
        self._layer_len = layer_len
        self._tf_setup()

    def fit(self, x_train, y_train):
        '''Train the neural network.'''
        init = tf.global_variables_initializer()
        self.sess.run(init)
        if self.batch_size > x_train.shape[0]:
            print('changing batch size to ', x_train.shape[0])
            self.batch_size = x_train.shape[0]
        y_batch = np.zeros([self.batch_size, self.box_size])

        picked_boxes, box_picks, _ = \
                _extract_boxes_around_picks(np.vstack(x_train[:, 0]),
                                            x_train[:, 1],
                                            self.box_size)

        for step in range(self.num_steps):
            if picked_boxes.shape[0] > self.batch_size:
                batch_start = (step * self.batch_size) \
                    % (picked_boxes.shape[0] - self.batch_size)
            else:
                batch_start = 0
            x_batch = picked_boxes[batch_start : batch_start + self.batch_size,
                                   :, np.newaxis]
            # Set y_batch to be zero except at the pick location. If the pick
            # falls between two points, linearly interpolate between them.
            # E.g. pick=1.5, y_batch=[0, 0.5, 0.5, 0, ...]
            y_batch[:, :] = 0
            for i in range(self.batch_size):
                pick_loc = box_picks[batch_start + i]
                remainder = pick_loc - int(pick_loc)
                y_batch[i, int(pick_loc)] = 1 - remainder
                y_batch[i, int(pick_loc) + 1] = remainder
            _, l = self.sess.run([self.optimizer, self.loss],
                                 feed_dict={self.box: x_batch,
                                            self.y_true: y_batch})
            if (self.print_interval is not None) \
                    and (step % self.print_interval == 0):
                print('step', step, l)

    def predict(self, x_predict):
        '''Use the neural network to predict picks.'''

        boxes_to_predict, _, box_starts = \
                _extract_boxes_around_picks(np.vstack(x_predict[:, 0]),
                                            x_predict[:, 1],
                                            self.box_size)
        raw_out = self.sess.run(self.layer1, \
                feed_dict={self.box: boxes_to_predict[:, :, np.newaxis]})
        # interpolate so we can pick with greater than integer accuracy
        interp_func = interp1d(np.arange(raw_out.shape[1]),
                               raw_out, kind='quadratic')
        interp_out = interp_func(np.linspace(0, raw_out.shape[1] - 1,
                                             self.num_interp))
        # find peak, undo interpolation, and 
        # shift picks to account for position of boxes
        return np.argmax(interp_out, axis=1) / (self.num_interp - 1) \
                * (raw_out.shape[1] - 1) + box_starts


def predict_picks(volume_data, frame_params, trace_params,
                  picks, frame_idxs_to_predict=None,
                  approx_reg=LinearRegression(),
                  fine_reg=Fine_reg_tensorflow(15, 5, 1000),
                  require_fit=True):
    '''Predict the picks for the specified frames.

    Arguments:
        volume_data: data to predict
        frame_params, trace_params: parameters for the data
        picks: list of known picks
        frame_idxs_to_predict: (optional) an integer or array of integers.
            If unset, predict all frames. If an integer, predict the
            frame with that index. If an array, predict all of the
            frame indices in the array.
        approx_reg, fine_reg: estimators to predict picks
        require_fit: boolean specifying whether to run fit methods

    Returns:
        An array of predicted picks.
    '''

    num_frames, frame_len, trace_len = volume_data.shape

    if frame_idxs_to_predict is None:
        # predict all frames
        frame_idxs_to_predict = np.arange(0, num_frames)

    if np.isscalar(frame_idxs_to_predict):
        # a single frame idx was supplied, but an array is expected
        frame_idxs_to_predict = np.array([frame_idxs_to_predict])

    picked_frame_idxs = [idx for idx, _ in enumerate(picks)
                         if len(picks[idx]) > 0]

    if len(picked_frame_idxs) == 0:
        # no traces have been picked, so cannot make prediction
        return []

    # arrange picks into a single array
    picks = [frame_picks for frame_picks in picks if len(frame_picks) > 0]
    picks = np.concatenate(picks)

    approx_picks = _apply_approx_reg_estimator(frame_params, trace_params,
                                               picks, frame_idxs_to_predict,
                                               picked_frame_idxs, trace_len,
                                               approx_reg, require_fit)
    fine_picks = _apply_fine_reg_estimator(volume_data,
                                           picks, approx_picks,
                                           frame_idxs_to_predict,
                                           picked_frame_idxs, fine_reg,
                                           require_fit)

    return fine_picks


def _apply_approx_reg_estimator(frame_params, trace_params,
                                picks, frame_idxs_to_predict,
                                picked_frame_idxs, trace_len,
                                approx_reg, require_fit):
    '''Use the frame and trace parameters to approximately predict picks.
    '''

    if require_fit:
      # extract data for picked traces
      picked_params = _extract_params(picked_frame_idxs,
                                      frame_params, trace_params)

      # train approx_reg estimator
      approx_reg.fit(picked_params, picks)

    # extract data for traces that are to have their picks predicted
    params_to_predict = _extract_params(frame_idxs_to_predict,
                                        frame_params, trace_params)

    # predict picks using approx_reg estimator
    approx_picks = approx_reg.predict(params_to_predict)

    # clip output so picks are within trace
    approx_picks = np.clip(approx_picks, 0, trace_len)

    return approx_picks


def _apply_fine_reg_estimator(volume_data,
                              picks, approx_picks, frame_idxs_to_predict,
                              picked_frame_idxs, fine_reg,
                              require_fit):
    '''Extract windows around the approximate picks and use another estimator
    to predict picks from these windows.
    '''

    if require_fit:
      # extract data for picked traces
      picked_volume_data = _extract_volume_data(picked_frame_idxs, volume_data)

      # combine volume data and picks into feature vectors
      picked_features = np.array([(picked_volume_data[i, :], picks[i])
                                  for i in range(picked_volume_data.shape[0])])

      # train approx_reg estimator
      fine_reg.fit(picked_features, picks)

    # extract data for traces that are to have their picks predicted
    volume_data_to_predict = _extract_volume_data(frame_idxs_to_predict,
                                                  volume_data)

    # combine volume data and picks into feature vectors
    features_to_predict = \
            np.array([(volume_data_to_predict[i, :], approx_picks[i])
                      for i in range(volume_data_to_predict.shape[0])])

    # predict picks using fine_reg estimator
    fine_picks = fine_reg.predict(features_to_predict)

    # clip output so picks are within trace
    trace_len = volume_data.shape[2]
    fine_picks = np.clip(fine_picks, 0, trace_len)

    return fine_picks


def _extract_boxes_around_picks(volume_data, picks, box_size):
    '''Extract a window of data around the pick for each trace.
 
    Returns:
      boxes: the windowed data
      picks_shifted: the location of the pick within the window
      box_starts: the starting index of each window
    '''
    half_box_size = int(box_size/2)
    picks_int = np.array(picks).astype(np.int)

    num_traces = len(picks)
    assert volume_data.shape[0] == num_traces

    boxes = np.zeros([num_traces, box_size])

    # pad volume_data in the trace dimension so boxes do not go out of bounds
    padded_volume_data = np.pad(volume_data,
                                ((0, 0), (half_box_size, half_box_size)),
                                'constant')

    # loop over traces and extract box
    # as padding has been applied, a box of volume_data centered on pick will be
    # a box from pick to pick + box_size of padded_volume_data
    for trace_idx in range(num_traces):
        boxes[trace_idx, :] = \
                padded_volume_data[trace_idx,
                                   picks_int[trace_idx]
                                   : picks_int[trace_idx] + box_size]

    # the estimator should find the picks are in the middle of the box
    picks_shifted = half_box_size + (picks - picks_int)

    # the start index of the boxes in volume_data (may be negative)
    box_starts = picks_int - half_box_size

    return boxes, picks_shifted, box_starts


def _extract_volume_data(frame_idxs, volume_data):
    '''Extract the data corresponding to the specified frame indices.'''
    extracted_volume_data = volume_data[frame_idxs, :, :]
    num_extracted_traces = np.prod(extracted_volume_data.shape[0:2])
    return extracted_volume_data.reshape(num_extracted_traces, -1)


def _extract_params(frame_idxs, frame_params, trace_params):
    '''Extract the frame and trace parameters corresponding to the specified
    frame indices.
    '''
    # frame parameters
    extracted_frame_params = frame_params[frame_idxs, :]
    # create a trace dimension so can repeat for each trace
    extracted_frame_params = \
            extracted_frame_params.reshape(extracted_frame_params.shape[0],
                                           1,
                                           extracted_frame_params.shape[1])
    # repeat so each trace has the frame parameter
    frame_len = trace_params.shape[1]
    extracted_frame_params = np.repeat(extracted_frame_params, frame_len, 1)

    # trace parameters
    extracted_trace_params = trace_params[frame_idxs, :, :]

    # combine trace and frame parameters into single set of parameters
    extracted_params = np.concatenate([extracted_trace_params,
                                       extracted_frame_params],
                                      axis=2)

    num_extracted_traces = np.prod(extracted_params.shape[0:2])
    return extracted_params.reshape(num_extracted_traces, -1)
