import numpy as np
import pytest
from sklearn.linear_model import LinearRegression
from pypick.predict_picks import (_apply_approx_reg_estimator,
                                  _apply_fine_reg_estimator,
                                  _extract_boxes_around_picks,
                                  predict_picks,
                                  Fine_reg_tensorflow)

FINE_ITS = 2000
np.random.seed(0)

@pytest.fixture
def one_trace():
    trace_len = 100
    picks = np.array([[50]])
    volume_data = np.zeros([1, 1, trace_len])
    volume_data[0, 0, picks[0][0]] = 1
    frame_params = np.zeros([1, 2])
    trace_params = np.zeros([1, 1, 1])
    return (volume_data, frame_params, trace_params, picks)


@pytest.fixture
def three_traces():
    num_traces = 3
    trace_len = 100
    picks = np.array([[50], [53], [56]])
    volume_data = np.zeros([num_traces, 1, trace_len])
    for trace_idx, pick in enumerate(picks):
        volume_data[trace_idx, 0, pick[0]] = 1
    frame_params = np.zeros([num_traces, 2])
    frame_params[:, 0] = np.arange(num_traces)
    trace_params = np.zeros([num_traces, 1, 1])
    return (volume_data, frame_params, trace_params, picks)


def test_one_trace_approx(one_trace):
    '''predict on trained trace, should give correct prediction.'''
    volume_data, frame_params, trace_params, picks = one_trace
    frame_idxs_to_predict = [0]
    picked_frame_idxs = [0]
    approx_reg = LinearRegression()
    trace_len = volume_data.shape[2]
    picks = np.concatenate(picks)
    approx_picks = _apply_approx_reg_estimator(frame_params, trace_params,
                                               picks, frame_idxs_to_predict,
                                               picked_frame_idxs, trace_len,
                                               approx_reg, True)
    assert np.allclose(approx_picks, picks)


def test_one_trace_fine(one_trace):
    '''predict on trained trace, should give correct prediction.'''
    volume_data, frame_params, trace_params, picks = one_trace
    frame_idxs_to_predict = [0]
    picked_frame_idxs = [0]
    box_size = 5
    layer_len = 3
    picks = np.concatenate(picks)
    approx_picks = picks.copy()
    fine_reg = Fine_reg_tensorflow(box_size, layer_len, FINE_ITS)
    fine_picks = _apply_fine_reg_estimator(volume_data,
                                           picks, approx_picks,
                                           frame_idxs_to_predict,
                                           picked_frame_idxs,
                                           fine_reg, True)
    assert np.allclose(fine_picks, picks, atol=1e-2)


def test_three_trace_approx(three_traces):
    '''predict on trained traces, should give correct prediction.'''
    volume_data, frame_params, trace_params, picks = three_traces
    frame_idxs_to_predict = [0, 1, 2]
    picked_frame_idxs = [0, 1, 2]
    approx_reg = LinearRegression()
    trace_len = volume_data.shape[2]
    picks = np.concatenate(picks)
    approx_picks = _apply_approx_reg_estimator(frame_params, trace_params,
                                               picks, frame_idxs_to_predict,
                                               picked_frame_idxs, trace_len,
                                               approx_reg, True)
    assert np.allclose(approx_picks, picks)


def test_three_trace_fine(three_traces):
    '''predict on trained traces, should give correct prediction.'''
    volume_data, frame_params, trace_params, picks = three_traces
    frame_idxs_to_predict = [0, 1, 2]
    picked_frame_idxs = [0, 1, 2]
    box_size = 5
    layer_len = 3
    picks = np.concatenate(picks)
    approx_picks = picks.copy()
    fine_reg = Fine_reg_tensorflow(box_size, layer_len, FINE_ITS)
    fine_picks = _apply_fine_reg_estimator(volume_data,
                                           picks, approx_picks,
                                           frame_idxs_to_predict,
                                           picked_frame_idxs,
                                           fine_reg, True)
    assert np.allclose(fine_picks, picks, atol=1e-2)


def test_three_trace_approx_2(three_traces):
    '''train on two traces, predict third.'''
    volume_data, frame_params, trace_params, picks = three_traces
    frame_idxs_to_predict = [2]
    picked_frame_idxs = [0, 1]
    approx_reg = LinearRegression()
    trace_len = volume_data.shape[2]
    train_picks = np.concatenate(picks[:2])
    approx_picks = _apply_approx_reg_estimator(frame_params, trace_params,
                                               train_picks,
                                               frame_idxs_to_predict,
                                               picked_frame_idxs, trace_len,
                                               approx_reg, True)
    assert np.allclose(approx_picks, picks[frame_idxs_to_predict])


def test_three_trace_fine_2(three_traces):
    '''train on two traces, predict third.'''
    volume_data, frame_params, trace_params, picks = three_traces
    frame_idxs_to_predict = [2]
    picked_frame_idxs = [0, 1]
    box_size = 5
    layer_len = 3
    train_picks = np.concatenate(picks[picked_frame_idxs])
    approx_picks = np.concatenate(picks[frame_idxs_to_predict])
    print(train_picks, approx_picks)
    fine_reg = Fine_reg_tensorflow(box_size, layer_len, FINE_ITS)
    fine_picks = _apply_fine_reg_estimator(volume_data,
                                           train_picks, approx_picks,
                                           frame_idxs_to_predict,
                                           picked_frame_idxs,
                                           fine_reg, True)
    assert np.allclose(fine_picks, picks[frame_idxs_to_predict], atol=1e-2)
