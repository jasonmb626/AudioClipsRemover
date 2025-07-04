import sys
import os.path

import soundfile as sf # You'll need to install this library: pip install soundfile
import music_tag
import numpy as np
import scipy

MATCH_THRESHOLD_DEFAULT = 1
CORRELATION_CHUNK_MINUTES_DEFAULT = 30

try:
    os.mkdir('trimmed')
    os.mkdir('discard')
except:
    pass

class Tee():
    """A basic logger that allows logging to console and optionally to a logfile as well."""
    def __init__(self, logpath):
        self.logfile = None
        if logpath:
            self.logfile = open(logpath, 'w')
    
    def log(self, message):
        print(message)
        if self.logfile:
            self.logfile.write(message + '\n')

    def __del__(self):
        if self.logfile:
            self.logfile.close()


class BaseAudioData():
    """Base class, not intended to be instantiated. 
    1. Holds audio data and information about audio data.
    2. Automatically converts stereo tracks to mono on load if necessary
    3. simple method to remove a range and optionally add said removed range to discard."""
    def __init__(self, audio_data, audio_samplerate, logger=None):
        self.discard_data = None
        self.audio_data = audio_data
        self.audio_samplerate = audio_samplerate
        if audio_data.ndim > 1:
            right_side = self.audio_data[:,0]
            left_side = self.audio_data[:,1]
            mono_data = (right_side + left_side) / 2
            del self.audio_data #Might not be true but there's a chance explicitly calling del this way speeds up how quickly garbage collection kicks in
            self.audio_data = mono_data
        self.audio_mean = np.mean(np.abs(self.audio_data))
        self.original_data_clip_len = self.audio_data.size
        self.logger = logger
        
    def remove_audio_range(self, start_index, end_index, capture_discard=False):
        if capture_discard:
            self.discard_data = np.append(self.audio_data[start_index:end_index], self.discard_data)    
        self.audio_data = np.delete(self.audio_data, slice(start_index, end_index))
             

class UnwantedClip(BaseAudioData):
    """Holds the data for an unwanted clip that should be removed
    1. Automatically trims loaded content to ignore leading and trailing silence
    2. Keeps original clip lenght in memory as this is how much the AudioClipsRemover class will actually remove"""
    def __init__(self, audio_data, audio_samplerate, logger=None, match_threshold=MATCH_THRESHOLD_DEFAULT):
        super().__init__(audio_data, audio_samplerate, logger)
        self.trimmed_clip_start = 0
        self.trimmed_clip_end = 0
        self._trim_data_clip()
        self.match_threshold = match_threshold
   
    def _trim_data_clip(self):
        i = 0
        cnt_negligible = 0
        while cnt_negligible <= 50:
            if abs(self.audio_data[i]) <= 1e-1:
                cnt_negligible = 0
            else:
                cnt_negligible += 1
            i += 1
        self.trimmed_clip_start = i - cnt_negligible
        self.remove_audio_range(0, self.trimmed_clip_start)
        i = self.audio_data.size - 1
        cnt_negligible = 0
        while cnt_negligible <= 50:
            if abs(self.audio_data[i]) <= 1e-1:
                cnt_negligible = 0
            else:
                cnt_negligible += 1
            i -= 1
        self.trimmed_clip_end = i + cnt_negligible
        self.remove_audio_range(self.trimmed_clip_end, self.audio_data.size)
        self.audio_mean = np.mean(np.abs(self.audio_data))


class AudioClipsRemover(BaseAudioData):
    """Holds the original (whole) audio that should be checked for clips to remove
    Also holds a list of unwanted data clips against which to check"""

    @staticmethod
    def match_tags(orig_file, match_file, logger=None):
        """After trimmed audio is saved to audio file, match the tags to the original untrimmed audio file"""
        orig = music_tag.load_file(orig_file)
        new = music_tag.load_file(match_file)
        for key in orig._TAG_MAP.keys():
            if logger:
                logger.log('Setting ' + key)
            if key == '#codec':
                continue
            elif key == 'totaltracks':
                new[key] = 0
            elif key == 'discnumber':
                new[key] = 0
            elif key == 'totaldiscs':
                new[key] = 0
            elif key == 'year':
                #This library does not allow multiple years (throws exception) but some audio tags include multiple years, so just take the first one
                new[key] = str(orig['year']).split(',')[0] 
            else:
                new[key] = orig[key]
        new.save()
        if logger:
            logger.log (match_file + ' saved')
 
    def __init__(self, audio_data, audio_samplerate, logger=None, correlation_chunk_minutes=CORRELATION_CHUNK_MINUTES_DEFAULT):
        super().__init__(audio_data, audio_samplerate, logger)
        self.discard_data = np.array([])
        self._correlation = None
        self._audio_for_correlation = None
        self._unwanted_clips = []
        #Correlations take longer to calculate the longer the chunk being checked and it grows exponentially by size.
        #Split original (whole) audio into manageable chunks.
        #Consequences of not splitting include:
        #1. Longer processing time
        #2. bad alloc exceptions
        #3. (On Windows) -- blue screen of death -- seriously
        self._correlation_chunk_minutes = correlation_chunk_minutes
        self._longest_unwanted_clip_frames = 0
        self._unwanted_clip_ranges = []

    def add_unwanted_clip(self, unwanted_clip: UnwantedClip):
        if unwanted_clip.audio_samplerate != self.audio_samplerate:
            raise Exception('Sample rates do not match.')
        self._unwanted_clips.append(unwanted_clip)
        if unwanted_clip.original_data_clip_len > self._longest_unwanted_clip_frames:
            self._longest_unwanted_clip_frames = unwanted_clip.original_data_clip_len
                
    def save_audio(self, outfilepath):
        """Helper method to save the (hopefully) trimmed audio to a new file."""
        sf.write(outfilepath, self.audio_data, self.audio_samplerate)
        if self.logger:
            self.logger.log(outfilepath + ' trimmed audio saved.')
        
    def save_discard(self, outfilepath):
        """Helper method to save the discarded audio (if any) to a new file."""
        if self.discard_data.size > 0:
            sf.write(outfilepath, self.discard_data, self.audio_samplerate)
            if self.logger:
                self.logger.log(outfilepath + ' discard audio saved.')
        else:
            if self.logger:
                self.logger.log('No discarded audio present, so not saving.')

    def _correlate_with_sample(self, sample : UnwantedClip, start_frame: int, num_frames: int):
        """Determine end frame, get a copy of the audio chunk against which to perform the correlation, and then do said correlation"""
        # Ensure both audio files have the same sample rate (adjust if necessary)
        if sample.audio_samplerate != self.audio_samplerate:
            raise Exception('Sample rates do not match.')
        if self._correlation is not None:
            del self._correlation #Might not be true but there's a chance explicitly calling del this way speeds up how quickly garbage collection kicks in
        if self._audio_for_correlation is not None:
            del self._audio_for_correlation #Might not be true but there's a chance explicitly calling del this way speeds up how quickly garbage collection kicks in
        end_frame = start_frame + num_frames
        if end_frame > self.audio_data.size:
            end_frame = self.audio_data.size
        self._audio_for_correlation = self.audio_data[start_frame:end_frame].copy()
        if self.logger:
            self.logger.log('Doing correlation on ' + str(self._audio_for_correlation.size) + ' frames.')
        self._correlation = scipy.signal.correlate(self._audio_for_correlation, sample.audio_data, mode='full')
        if self.logger:
            self.logger.log('Did correlation. Len=' + str(self._correlation.size))
        
    def _set_peak_correlation(self, sample: UnwantedClip, start_frame: int):
        """Sets peak correlation and offsets for use in other methods
        This method probably needs work as it's not completely logical. Maybe it should return the data instead of setting it at object-level"""
        self.moved_offset = 0
        self.peak_index = np.argmax(self._correlation)    
        self.check_offset_samples = self.peak_index + start_frame - (sample.audio_data.size - 1) 
        self.offset_samples = self.peak_index + start_frame - (sample.audio_data.size + sample.trimmed_clip_start - 1)
        if self.offset_samples < 0:
            self.moved_offset = -self.offset_samples
            if self.logger:
                self.logger.log('Moving offset_samples from ' + str(self.offset_samples) + ' to 0.')
                self.logger.log('self.moved_offset=' + str(self.moved_offset))
            self.offset_samples = 0

    def _calculate_correlation_accuracy(self, sample: UnwantedClip):
        """_set_peak_correlation will simply set the index most likely to be an unwanted data clip. It does not mean that index actually contains said unwanted data.
        This method does a simple check of the matched data against the sample
        I think there is probably a better way to determine accuracy of correlation but so far haven't found one.
        Other things I tried:
        1. Making sense of the value at peak_index, but it seems to be all relative
        2. Noting how quickly (shape of curve) the correlation matches on either side of the one at peak index, but couldn't find any real meaning there either."""
        matched_data = self.audio_data[self.check_offset_samples:self.check_offset_samples + sample.audio_data.size]
        mean = np.mean(np.abs(matched_data))
        matched_data = matched_data * sample.audio_mean / mean #Adjust for audio volume differences
        diff = np.sum(matched_data - sample.audio_data)
        if self.logger:
            self.logger.log('Calculated difference: ' + str(diff) + '. correlation=' + str(self._correlation[self.peak_index]))
        return diff
    
    def find_unwanted_clip_ranges(self):
        """This method splits the original (whole) audio into smaller chunks, performs a correlation against all unwanted audio clips, determines accuracy, and populates
        a list of ranges to cut if those ranges are within the match threshold
        Perhaps it should return the list instead? Or have a different name."""
        start_frame = 0
        end_frame = 0
        frames_to_check = self._correlation_chunk_minutes * 60 * self.audio_samplerate
        while end_frame < self.audio_data.size:
            start_frame = end_frame - self._longest_unwanted_clip_frames - 1
            if start_frame < 0:
                start_frame = 0
            end_frame = start_frame + frames_to_check
            if end_frame > self.audio_data.size:
                end_frame = self.audio_data.size
            for unwanted_clip in self._unwanted_clips:
                self._correlate_with_sample(unwanted_clip, start_frame, frames_to_check)
                rounds = 0
                while True: #Get peak correlation and then test its accuracy in a loop until the accuracy is poor
                    diff = unwanted_clip.match_threshold
                    rounds += 1
                    # Find the index of the maximum correlation
                    self.logger.log('Round ' + str(rounds) + ' getting peak correlation.')
                    self._set_peak_correlation(unwanted_clip, start_frame)
                    if self.check_offset_samples < 0:
                        break
                    diff = self._calculate_correlation_accuracy(unwanted_clip)      
                    if abs(diff) < unwanted_clip.match_threshold:
                        if self.logger:
                            self.logger.log('Found unwanted clip at ' + str(self.offset_samples) + ' - ' + str(self.offset_samples / self.audio_samplerate))
                        self._unwanted_clip_ranges.append((self.offset_samples, self.offset_samples + unwanted_clip.original_data_clip_len))
                        #Zero out the range we already confirmed, so it won't show as peak correlation on next round
                        self._correlation[self.offset_samples - start_frame:self.offset_samples - start_frame + unwanted_clip.original_data_clip_len] = 0
                    else:
                        break

    def remove_found_unwanted_clip_ranges(self, capture_discard=False):
        """Sorts the unwanted clip ranges as set by find_unwanted_clip_ranges in descending order
        Descending order so that the ranges can be deleted without affecting the offset of the next range to be deleted.
        Deletes aid ranges, allowing for capture of discard data"""
        self._unwanted_clip_ranges.sort(key = lambda x: x[0], reverse=True)
        for unwanted_clip_range in self._unwanted_clip_ranges:
            start_of_range, end_of_range = unwanted_clip_range
            self.remove_audio_range(start_of_range, end_of_range, capture_discard)
            if self.logger:
                self.logger.log('Removed from ' + str(start_of_range) + ' to ' + str(end_of_range))
            
    def print_found_unwanted_clip_ranges(self):
        """For debugging, print the ranges that would be deleted by remove_found_unwanted_clip_ranges"""
        self._unwanted_clip_ranges.sort(key = lambda x: x[0], reverse=True)
        for unwanted_clip_range in self._unwanted_clip_ranges:
            start_frame, end_frame = unwanted_clip_range
            print('Start: ' + str(start_frame / self.audio_samplerate) + ' End: ' + str(end_frame / self.audio_samplerate))


def make_audio_clips_remover(data):
    """Instantiates an AudioClipsRemover class with all data populated so all we need to do is call the find, remove, save methods in main"""
    """Input is a dict generated from command-line args and other defaults"""
    logger = Tee(data['log_filepath'])
    podcast_data, podcast_samplerate = sf.read(data['original_whole_audio'])
    podcast = AudioClipsRemover(podcast_data, podcast_samplerate, logger)
    for unwanted_clip_path in data['unwanted_clip_paths']:
        if unwanted_clip_path.endswith('.npy'):
            unwanted_clip_data = np.load(unwanted_clip_path)
            unwanted_clip_samplerate = podcast_samplerate
        else:
            unwanted_clip_data, unwanted_clip_samplerate = sf.read(unwanted_clip_path)
        unwanted_clip = UnwantedClip(unwanted_clip_data, unwanted_clip_samplerate, logger)
        podcast.add_unwanted_clip(unwanted_clip)
    return podcast

def process_args_to_dict(args):
    """Convert command-line args to a dict that can be used by make_audio_clips_remover"""
    data = {}
    data['unwanted_clip_paths'] = []
    while len(args) > 0:
        if args[0] == '-u':
            args.pop(0)
            data['unwanted_clip_paths'].append(args[0])
            args.pop(0)
        else:
            data['original_whole_audio'] = args[0]
            args.pop(0)        
    data['trimmed_podcast_filepath'] = os.path.join('trimmed', data['original_whole_audio'])
    data['discard_filepath'] = os.path.join('discard', data['original_whole_audio'])
    data['log_filepath'] = os.path.join('trimmed', data['original_whole_audio'].replace(".mp3", ".txt"))
    return data
            
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Missing arguments')
        sys.exit()

    arg_dict = process_args_to_dict(sys.argv)
    audio_clips_remover = make_audio_clips_remover(arg_dict)
    audio_clips_remover.find_unwanted_clip_ranges()
    audio_clips_remover.remove_found_unwanted_clip_ranges(capture_discard=True)
    audio_clips_remover.save_audio(arg_dict['trimmed_podcast_filepath'])
    audio_clips_remover.save_discard(arg_dict['discard_filepath'])
    AudioClipsRemover.match_tags(arg_dict['original_whole_audio'], arg_dict['trimmed_podcast_filepath'], audio_clips_remover.logger)