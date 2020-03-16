import numpy as np

from td_utils import *

# Global Variables                                                                

# the number of time steps input to sequence model from spectrogram               
Tx = 5511
# number of frequencies input to the model at each time step of the spectrogram   
n_freq = 101

Ty = 1375 # number of time steps in the output of our model                       



def get_random_time_segment(segment_ms):
    """
    Gets a random time segment of duration segment_ms in a 10,000 ms audio clip.
    
    Arguments:
    segment_ms -- the duration of the audio clip in ms 
    ("ms" stands for "milliseconds")
    
    Returns:
    segment_time -- a tuple of (segment_start, segment_end) in ms
    """
    
    segment_start = np.random.randint(low=0, high=10000-segment_ms)   
        # Make sure segment doesn't run past the 10sec background 
    segment_end = segment_start + segment_ms - 1
    
    return (segment_start, segment_end)


def is_overlapping(segment_time, previous_segments):
    """
    Checks if the time of a segment overlaps with the times of existing segments.
    
    Arguments:
    segment_time -- a tuple of (segment_start, segment_end) for the new segment
    previous_segments -- a list of tuples of (segment_start, segment_end) 
     for the existing segments
    
    Returns:
    True if the time segment overlaps with any of the existing segments, 
    False otherwise
    """
    
    segment_start, segment_end = segment_time
    

    overlap = False
    
    # loop over the previous_segments start and end times.
    # Compare start/end times and set the flag to True if there is an overlap

    for previous_start, previous_end in previous_segments:
        if (segment_start <= previous_end) & (segment_end >= previous_start):
            overlap = True


    return overlap



def insert_audio_clip(background, audio_clip, previous_segments):
    """
    Insert a new audio segment over the background noise 
    at a random time step, ensuring that the 
    audio segment does not overlap with existing segments.
    
    Arguments:
    background -- a 10 second background audio recording.  
    audio_clip -- the audio clip to be inserted/overlaid. 
    previous_segments -- times where audio segments have already been placed
    
    Returns:
    new_background -- the updated background audio
    """
    
    # Get the duration of the audio clip in ms
    segment_ms = len(audio_clip)
    
    segment_time = get_random_time_segment(segment_ms)
    
    while is_overlapping(segment_time, previous_segments):
        segment_time = get_random_time_segment(segment_ms)

    previous_segments.append(segment_time)
    
    # Superpose audio segment and background
    new_background = background.overlay(audio_clip, position = segment_time[0])
    
    return new_background, segment_time



def insert_ones(y, segment_end_ms):
    """
    Update the label vector y. The labels of the 50 output steps 
    strictly after the end of the segment should be set to 1. 
    By strictly we mean that the label of segment_end_y should be 0 while, the
    50 following labels should be ones.
    
    
    Arguments:
    y -- numpy array of shape (1, Ty), the labels of the training example
    segment_end_ms -- the end time of the segment in ms
    
    Returns:
    y -- updated labels
    """
    
    # duration of the background (in terms of model output time-steps)
    segment_end_y = int(segment_end_ms * Ty / 10000.0)
    
    # Add 1 to the correct index in the background label (y)
    for i in range(segment_end_y+1, segment_end_y+51):
        if i < Ty:
            y[0, i] = 1
    
    return y


def create_training_example(background, activates, negatives, e=1):
    """
    Creates a training example with a given background, activates, and negatives.
    
    Arguments:
    background -- a 10 second background audio recording
    activates -- a list of audio segments of the word "heyjoe"
    negatives -- a list of audio segments of random words that are not "heyjoe"

    e -- counter that is helpful to add to "train.wav" at the end to save different
    training examples with different names as in "train1.wav" or "train20.wav"
    Currently unused but I use it occasionally for debugging.
    
    Returns:
    x -- the spectrogram of the training example
    y -- the label at each time step of the spectrogram
    """
    
    # Make background quieter
    background = background - 20


    # Initialize y (label vector) with zeros
    y = np.zeros((1, Ty))

    # Initialize segment times as an empty list
    previous_segments = []
    
    
    # Select 0-4 random "activate" audio clips from the entire list of "activates" recordings
    number_of_activates = np.random.randint(0, 3)
    random_indices = np.random.randint(len(activates), size=number_of_activates)
    random_activates = [activates[i] for i in random_indices]
    

    # Loop over randomly selected "activate" clips and insert in background
    for random_activate in random_activates:
        # Insert the audio clip on the background
        background, segment_time = insert_audio_clip(background, random_activate, previous_segments)
        # Retrieve segment_start and segment_end from segment_time
        segment_start, segment_end = segment_time
        # Insert labels in "y"
        y = insert_ones(y, segment_end)


    # Select 0-2 random negatives audio recordings from the entire list of "negatives" recordings
    if number_of_activates == 2:
        number_of_negatives = np.random.randint(0, 1)
    else:
        number_of_negatives = np.random.randint(0, 3)   ## Avoid chance of collisions

    print("Activates ", number_of_activates, "Negatives ", number_of_negatives)

    random_indices = np.random.randint(len(negatives), size=number_of_negatives)
    random_negatives = [negatives[i] for i in random_indices]


    # Loop over randomly selected negative clips and insert in background
    for random_negative in random_negatives:
        # Insert the audio clip on the background 
        background, _ = insert_audio_clip(background, random_negative, previous_segments)


    
    # Standardize the volume of the audio clip 
    background = match_target_amplitude(background, -20.0)

    # Export new training example 
    file_handle = background.export("train" +  ".wav", format="wav")
    print("File (train.wav) was saved in your directory.")
    
    # Get and plot spectrogram of the new recording (background with superposition of positive and negatives)
    x = graph_spectrogram("train.wav")
    
    return x, y



