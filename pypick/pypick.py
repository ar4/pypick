'''Pick events in multiple frames of data using Matplotlib and machine
learning.'''
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate
from sklearn.preprocessing import scale
from sklearn.svm import SVR
import pypick.predict_picks

class Pypicks(object):
    '''An object to allow picking and pick prediction on a data volume.'''
    def __init__(self, volume_data, frame_params, trace_params, picks=None,
                 approx_reg=SVR(C=100),
                 fine_reg=pypick.predict_picks.Fine_reg_tensorflow(15, 5, 1000),
                 keys=None, perform_prediction=True):
        '''Arguments:
            volume_data: the data to be picked (frames x traces x trace)
            frame_params: parameters associated with each frame
            trace_params: parameters associated with each trace
            picks: (optional) a list with one entry for each frame with
                initial picks. Frames with no picks can be empty.
            approx_reg: (optional) a regression object with fit and predict
                methods. If unset, a default method will be used.
            fine_reg: (optional) a regression object with fit and predict
                methods. If unset, a default method will be used.
            keys: (optional) a dictionary specifying which keys to use for
                different actions during picking. In unset, default keys
                will be used.
        '''
        self.num_frames, self.frame_len, self.trace_len = volume_data.shape
        assert frame_params.shape[0] == self.num_frames
        assert trace_params.shape[0:2] == (self.num_frames, self.frame_len)
        if picks is None:
            picks = [[] for frame in range(self.num_frames)]
        assert len(picks) == self.num_frames
        self.volume_data = volume_data
        self.frame_params = frame_params
        self.trace_params = trace_params
        self.picks = picks
        self.approx_reg = approx_reg
        self.fine_reg = fine_reg
        if keys is None:
            keys = {'set': 'a',
                    'exitsave': 'q',
                    'exitnosave': 'x',
                    'delete': 'd'}
        self.keys = keys
        self.perform_prediction = perform_prediction
        self.require_fit = True

        def _normalise_data(frame_params, trace_params):
            '''Scales frame and trace parameters as this is required by
            many machine learning methods.
            '''
            frame_params = scale(frame_params)
            orig_shape = trace_params.shape
            num_traces = np.prod(trace_params.shape[:2])
            trace_params = trace_params.reshape(num_traces, -1)
            trace_params = scale(trace_params)
            trace_params = trace_params.reshape(orig_shape)
            return frame_params, trace_params

        self.frame_params, self.trace_params = \
                _normalise_data(self.frame_params, self.trace_params)

    def pypick(self, frame_idx_to_pick=None, ax=None):
        '''Use PyPlot to allow user to pick events.

        Arguments:
            frame_idx_to_picks: (optional) if None, a frame selection screen
                will be shown. If set, the specified frame will be opened
                for picking.
            ax: (optional) the PyPlot axis to use. If None, a new axis will
                be created.
        '''

        def _select2d(points_x, points_y, points_color=None, ax=None):
            '''Display a 2D scatter plot with each point representing a frame
            that the user can click on to select which frame to pick. Picked
            frames are displayed in a different color.
            '''

            if ax is None:
                _, ax = plt.subplots()
            fig = ax.get_figure()

            ax.scatter(points_x, points_y, c=points_color, picker=True)

            def onpick(event):
                '''A frame has been selected, rerun pypick specifying that
                that frame index is to be picked.
                '''
                fig.canvas.mpl_disconnect(cig)
                ax.clear()
                self.pypick(frame_idx_to_pick=event.ind[0], ax=ax)

            cig = fig.canvas.mpl_connect('pick_event', onpick)

        def _pick_line(frame_data, picks, ax=None):
            '''Display a frame of data with a line showing current picks
            and allowing the user to adjust the line and save picks.
            '''

            class PickLine(object):
                '''The line of picks.'''
                def __init__(self, line, setkey):
                    '''Arguments:
                        line: a PyPlot line object representing the picks
                        setkey: a character containing the key that triggers
                            the 'set' action
                    '''
                    self.line = line
                    self.xs = np.array(self.line.get_xdata(), np.int)
                    self.ys = np.array(self.line.get_ydata(), np.float)
                    self.maxx = np.max(self.xs)
                    self.setkey = setkey

                def onclick(self, event):
                    '''The user has clicked. If the set key is also pressed,
                    update the pick value for the clicked location.
                    '''
                    if (event.button == 1) and (event.key == self.setkey):
                        x = int(np.round(event.xdata))
                        y = event.ydata
                        x = np.clip(x, 0, self.maxx)
                        try:
                            # there is already a pick at this 'x': update it
                            idx = np.nonzero(self.xs == x)[0][0]
                            self.ys[idx] = y
                        except:
                            # there is no previous pick for this 'x': create one
                            self.xs = np.append(self.xs, x)
                            self.ys = np.append(self.ys, y)
                            idxs = np.argsort(self.xs)
                            self.xs = self.xs[idxs]
                            self.ys = self.ys[idxs]
                        self.line.set_data(self.xs, self.ys)

            def finish_pick(event):
                '''Exit the picking screen and return to the frame choosing
                screen.
                '''

                def clear_plot():
                    '''Disconnect event triggers, clear axis, and return to
                    frame choosing screen.
                    '''
                    for cig in cigs:
                        fig.canvas.mpl_disconnect(cig)
                    ax.clear()
                    self.pypick(ax=ax)

                if event.key == self.keys['exitsave']:
                    if len(picker.xs) < 2:
                        print('must have at least 2 points')
                        return
                    interpolater = \
                            scipy.interpolate.interp1d(picker.xs, picker.ys,
                                                       fill_value='extrapolate')
                    x = np.arange(0, num_traces)
                    interpolated_picks = interpolater(x)
                    # save interpolated picks
                    self.picks[frame_idx_to_pick] = interpolated_picks
                    self.require_fit = True
                    clear_plot()

                if event.key == self.keys['exitnosave']:
                    clear_plot()

            def onselect(eclick, erelease):
                '''Delete picks within rectangle selector.'''
                start_point = (eclick.xdata, eclick.ydata)
                end_point = (erelease.xdata, erelease.ydata)
                del_idxs = np.where((picker.xs >= start_point[0])
                                    & (picker.xs <= end_point[0])
                                    & (picker.ys >= start_point[1])
                                    & (picker.ys <= end_point[1]))
                picker.xs = np.delete(picker.xs, del_idxs)
                picker.ys = np.delete(picker.ys, del_idxs)
                picker.line.set_data(picker.xs, picker.ys)

            def del_select_on(event):
                '''Turn on rectangular selector if key pressed.'''
                if event.key == self.keys['delete'] and not del_selector.active:
                    del_selector.set_active(True)

            def del_select_off(event):
                '''Turn off rectangular selector if key released.'''
                if event.key == self.keys['delete'] and del_selector.active:
                    del_selector.set_active(False)

            # beginning of _pick_line code
            num_traces = frame_data.shape[0]
            trace_len = frame_data.shape[1]

            if ax is None:
                _, ax = plt.subplots()
            fig = ax.get_figure()

            ax.imshow(frame_data.T, aspect='auto', cmap='gray')

            if len(picks) == 0:
                # No initial picks for this frame, so predict a straight
                # line across the middle of the frame
                initial_depth = (trace_len - 1) / 2
                picks = ([0, num_traces - 1],
                         [initial_depth, initial_depth])

            if len(picks) == 2:
                # Initial frame, so only have the two default picks
                line = ax.plot(picks[0], picks[1])
            else:
                line = ax.plot(picks)
            picker = PickLine(line[0], self.keys['set'])
            del_selector = matplotlib.widgets.RectangleSelector(ax, onselect)
            del_selector.set_active(False)

            cigs = []
            cigs.append(fig.canvas.mpl_connect('button_press_event',
                                               picker.onclick))
            cigs.append(fig.canvas.mpl_connect('key_press_event',
                                               finish_pick))
            del_select_on.del_selector = del_selector
            cigs.append(fig.canvas.mpl_connect('key_press_event',
                                               del_select_on))
            del_select_off.del_selector = del_selector
            cigs.append(fig.canvas.mpl_connect('key_release_event',
                                               del_select_off))

        #beginning of pypick code
        if ax is None:
            _, ax = plt.subplots()
        if frame_idx_to_pick is None:
            picked_frames = [len(x) > 0 for x in self.picks]
            _select2d(self.frame_params[:, 0], self.frame_params[:, 1],
                      points_color=picked_frames, ax=ax)
            return

        predicted_picks = self.predict(frame_idx_to_pick)
        _pick_line(self.volume_data[frame_idx_to_pick, :, :],
                   picks=predicted_picks, ax=ax)

    def predict(self, frame_idxs_to_predict=None):
        '''Predict the picks for the specified frames.

        Arguments:
            frame_idxs_to_predict: (optional) an integer or array of integers.
                If unset, predict all frames. If an integer, predict the
                frame with that index. If an array, predict all of the
                frame indices in the array.

        Returns:
            An array of predicted picks.
        '''
        if not self.perform_prediction:
            #user has requested no prediction
            if np.isscalar(frame_idxs_to_predict):
                #a single frame idx was supplied, return existing picks for it
                return self.picks[frame_idxs_to_predict]
            return []

        predicted_picks = \
                pypick.predict_picks.predict_picks(self.volume_data,
                                                   self.frame_params,
                                                   self.trace_params,
                                                   self.picks,
                                                   frame_idxs_to_predict,
                                                   approx_reg=self.approx_reg,
                                                   fine_reg=self.fine_reg,
                                                   require_fit=self.require_fit)
        self.require_fit = False

        return predicted_picks
