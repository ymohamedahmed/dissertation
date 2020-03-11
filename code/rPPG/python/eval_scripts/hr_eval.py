def evaluate_hr(rppg_hr, ecg_hr, number_of_frames, window_size, step_size):
    correct = []
    for index, hr in enumerate(rppg_hr):
        progress = window_size + (index*step_size)
        progress = int(len(ecg_hr)*progress/number_of_frames)
        start = int((len(ecg_hr)*index*step_size)/number_of_frames)
        reference = ecg_hr[start:progress]
        correct_ref = np.min(reference) <= hr and np.max(reference) >= hr
        correct.append(correct_ref)
    return correct

def evaluate_hr(rppg_hr, ecg_hr, number_of_frames, window_size, step_size):
    correct = []
    for index, hr in enumerate(rppg_hr):
        progress = window_size + (index*step_size)
        progress = int(len(ecg_hr)*progress/number_of_frames)
        start = int((len(ecg_hr)*index*step_size)/number_of_frames)
        reference = ecg_hr[start:progress]
        correct_ref = np.min(reference) <= hr and np.max(reference) >= hr
        correct.append(correct_ref)
    return correct



