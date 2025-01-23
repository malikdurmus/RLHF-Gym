from rlhf.buffers import PreferenceBuffer


def surf(reward_models, replay_buffer, human_feedback_pb):
    # ssl
    unlabeled_pb = PreferenceBuffer(human_feedback_pb.buffer_size)

    # tda
    return unlabeled_pb
def semi_supervised_labeling(reward_models,replay_buffer):
    #
    pass

