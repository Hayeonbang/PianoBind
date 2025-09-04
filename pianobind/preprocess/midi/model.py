import numpy as np
import pickle
from tqdm import tqdm
from .utils import Event, read_items, quantize_items, group_items, item2event, get_ticks_to_seconds_grid
from .utils import DEFAULT_VELOCITY_BINS, DEFAULT_FRACTION, DEFAULT_DURATION_BINS, DEFAULT_TEMPO_INTERVALS, DEFAULT_RESOLUTION, PITCH_MIN, PITCH_MAX
import miditoolkit
import copy
from pathlib import Path

Composer = {
    "Bethel": 0,
    "Clayderman": 1,
    "Einaudi": 2,
    "Hancock": 3,
    "Hillsong": 4,
    "Hisaishi": 5,
    "Ryuichi": 6,
    "Yiruma": 7,
    "Padding": 8,
}

Emotion = {
    "Q1": 0,
    "Q2": 1,
    "Q3": 2,
    "Q4": 3,
}


BAR = 0
BAR_NEW = 0

TRANSPOSE_VALS = range(-5, 7) # -5~6

class CP(object):
    def __init__(self, dict):
        # load dictionary
        self.event2word, self.word2event = pickle.load(open(dict, 'rb'))
        # pad word: ['Bar <PAD>', 'Position <PAD>', 'Pitch <PAD>', 'Duration <PAD>']
        self.pad_word = [self.event2word[etype]['%s <PAD>' % etype] for etype in self.event2word]

    def extract_events(self, input_path, task):
        note_items, tempo_items = read_items(input_path)
        if len(note_items) == 0:   # if the midi contains nothing
            return None
        note_items = quantize_items(note_items)
        max_time = note_items[-1].end
        items = tempo_items + note_items
        
        groups = group_items(items, max_time)
        events = item2event(groups, task)
        return events

    def padding(self, data, max_len, ans):
        pad_len = max_len - len(data)
        for _ in range(pad_len):
            if not ans:
                data.append(self.pad_word)
            else:
                data.append(0)

        return data

    def events_to_words(self, events, task):
        words, ys = [], []
        for note_tuple in events:
            nts, to_class = [], -1
            for e in note_tuple:
                e_text = '{} {}'.format(e.name, e.value)
                nts.append(self.event2word[e.name][e_text])
                if e.name == 'Pitch':
                    to_class = e.Type
            words.append(nts)
            if task == 'melody' or task == 'velocity':
                ys.append(to_class+1)
        return words, ys

    def slice_words_and_ys(self, words, ys, path, max_len, task):
        slice_words, slice_ys = [], []
        for i in range(0, len(words), max_len):
            slice_words.append(words[i:i+max_len])
            if task == "composer":
                name = path.split('/')[-2]
                slice_ys.append(Composer[name])
            elif task == "emotion":
                name = path.split('/')[-1].split('_')[0]
                slice_ys.append(Emotion[name])
            else:
                slice_ys.append(ys[i:i+max_len])
        
        # padding or drop
        # drop only when the task is 'composer' and the data length < max_len//2
        if len(slice_words[-1]) < max_len:
            if task == 'composer' and len(slice_words[-1]) < max_len//2:
                slice_words.pop() # remove
                slice_ys.pop() # remove
                last_word_idx = max_len
            else:
                last_word_idx = len(slice_words[-1]) - 1
                slice_words[-1] = self.padding(slice_words[-1], max_len, ans=False)
        else:
            last_word_idx = len(slice_words[-1]) - 1
        if (task == 'melody' or task == 'velocity') and len(slice_ys[-1]) < max_len:
            slice_ys[-1] = self.padding(slice_ys[-1], max_len, ans=True)
            
        return slice_words, slice_ys, last_word_idx
    
    def get_bar_start_token_idxs(self, words):
        bar_start_token_idxs = []
        for i, word in enumerate(words):
            if word[BAR] == BAR_NEW:
                bar_start_token_idxs.append(i)
        return bar_start_token_idxs
    
    def slice_words_and_ys_bar(self, words, ys, path, max_len, n_seg_bar, task, start_slice_idx, do_drop=True, overlap=True):
        slice_words, slice_ys, last_word_idxs = [], [], []
        cum_bar = 0
        prev_bar = 0
        bar_start_token_idxs = self.get_bar_start_token_idxs(words)
        
        prev_seg_bar = 0
        current_seg_bar = n_seg_bar

        segment_bar = dict()
        i_seg = 0
        # for current_seg_bar in slicing_range:
        while(current_seg_bar < len(bar_start_token_idxs)):
            prev_seg_bar_tok_idx = bar_start_token_idxs[prev_seg_bar]
            current_seg_bar_tok_idx = bar_start_token_idxs[current_seg_bar]
            last_word_idx = current_seg_bar_tok_idx - prev_seg_bar_tok_idx
            assert last_word_idx <= max_len
            last_word_idxs.append(last_word_idx)
            segment_bar[start_slice_idx+i_seg] = {'start bar': prev_seg_bar, 'end bar': current_seg_bar}

            slice_words.append(words[prev_seg_bar_tok_idx:current_seg_bar_tok_idx])
            # legacy part
            if task == "composer":
                name = path.split('/')[-2]
                slice_ys.append(Composer[name])
            elif task == "emotion":
                name = path.split('/')[-1].split('_')[0]
                slice_ys.append(Emotion[name])
            else:
                slice_ys.append(ys[prev_seg_bar_tok_idx:current_seg_bar_tok_idx])
            
            if overlap:
                prev_seg_bar += (n_seg_bar // 2) 
                current_seg_bar += (n_seg_bar // 2)
            else:
                prev_seg_bar = current_seg_bar
                current_seg_bar += n_seg_bar
            assert abs(prev_seg_bar-current_seg_bar) == n_seg_bar
            i_seg += 1
            
        last_word_idxs = { start_slice_idx+i: last_word_idx for i, last_word_idx in enumerate(last_word_idxs)}
        
        last_segment_bar = n_seg_bar
        if not do_drop:
            if prev_seg_bar < len(bar_start_token_idxs): 
                last_segment_bar = len(bar_start_token_idxs) - prev_seg_bar
                if overlap:
                    thres = (n_seg_bar // 2)
                else:
                    thres = n_seg_bar
                
                if last_segment_bar < n_seg_bar: # equal condition shouldn't have the residual segment
                    prev_seg_bar_tok_idx = bar_start_token_idxs[prev_seg_bar]
                    slice_words.append(words[prev_seg_bar_tok_idx:])
                    segment_bar[start_slice_idx+i_seg] = {'start bar': prev_seg_bar, 'end bar': prev_seg_bar+last_segment_bar}
                    print(start_slice_idx+i_seg, prev_bar, prev_seg_bar+last_segment_bar)
                    # legacy part
                    if task == "composer":
                        name = path.split('/')[-2]
                        slice_ys.append(Composer[name])
                    elif task == "emotion":
                        name = path.split('/')[-1].split('_')[0]
                        slice_ys.append(Emotion[name])
                    else:
                        slice_ys.append(ys[current_seg_bar_tok_idx:])                
        
        #### we don't have any padding here
        
        return slice_words, slice_ys, last_word_idxs, last_segment_bar, segment_bar
                               

    
    def update_piece_segment_info(self, all_piece_info, events, segmented_events, all_words, slice_words, path, max_len, last_word_idx, last_word_idxs, last_segment_bar, tp_val, segment_bar=None):
        start_slice_idx = len(all_words)
        end_slice_idx = len(all_words + slice_words)-1
        
        cum_bar = 0
        prev_bar = 0
        if segment_bar is None:
            segment_bar = dict()
            for i, slice_word in enumerate(slice_words):
                for word in slice_word:
                    # BAR is 0
                    if word[BAR] == BAR_NEW:
                        cum_bar += 1
                segment_bar[start_slice_idx+i] = {'start bar': prev_bar, 'end bar': cum_bar}
                prev_bar = cum_bar
        
        if path not in all_piece_info.keys():
            all_piece_info[path] = {}
                
        all_piece_info[path][tp_val] = {'start slice idx': start_slice_idx, 'end slice idx': end_slice_idx, 'last word idx':last_word_idx, 'segment bar': segment_bar, 'last word idxs': last_word_idxs, 'last segment bar': last_segment_bar}

        event_cum_bar = 0
        start_event_idx = 0
        end_event_idx = 0
        
        #### FOR MIDI DECODING
        segmented_events.update({seg_idx: [] for seg_idx in segment_bar})
        bar_start_times = []
        bar_start_event_idx = []
        for event_idx, event in enumerate(events):
            assert event[0].name == 'Bar'
            
            if event[0].value=='New':
                bar_start_times.append(int(event[1].time))
                bar_start_event_idx.append(event_idx)
        all_piece_info[path]['bar_start_times'] = { i: bar_start_time for i, bar_start_time in enumerate(bar_start_times)}
        
        
        # all_piece_info[path]['bar_start_times_sec']
        midi_obj = miditoolkit.midi.parser.MidiFile(path)
        ticks_to_seconds = get_ticks_to_seconds_grid(midi_obj)
        all_piece_info[path]['bar_start_times_sec'] = { i: round(ticks_to_seconds[bar_start_time], 2) for i, bar_start_time in enumerate(bar_start_times)} 
        
        for seg_idx, value in segment_bar.items():
            start_bar, end_bar = value['start bar'], value['end bar']
            if start_bar == len(bar_start_event_idx):
                start_bar -= 1
            if end_bar == len(bar_start_event_idx):
                end_bar = -1
            start_bar_idx = bar_start_event_idx[start_bar]
            end_bar_idx = bar_start_event_idx[end_bar]
            segmented_events[seg_idx] = events[start_bar_idx:end_bar_idx]
        
    def pitch_transpose(self, events, transpose):
        remove_list = []
        for i, event in enumerate(events):
            assert event[2].name == 'Pitch'
            events[i][2].value += transpose    
            if events[i][2].value < PITCH_MIN or events[i][2].value > PITCH_MAX: 
                remove_list.append(i)
            
        # pitch range check
        for i in reversed(remove_list):        
            del events[i]
        return events

    def prepare_data(self, midi_paths, out_dir, task, segment_mode, do_transpose, max_len, n_seg_bar):
        all_words, all_ys = [], []
        all_piece_info = dict()
        segmented_events = dict()
        
        for path in tqdm(midi_paths):
            # extract events
            events = self.extract_events(path, task)
            if not events:  # if midi contains nothing
                print(f'skip {path} because it is empty')
                continue
            
            ### TODO: generate +6 -5 transpose cases    
            # file name should be segment_idx_{transposed_value}
            if do_transpose:
                # print('do transpose')
                transpose_vals = TRANSPOSE_VALS
            else:
                transpose_vals = [0]
                
                
            for tp_val in transpose_vals:
                tp_events = copy.deepcopy(events)            
                tp_events = self.pitch_transpose(tp_events, tp_val)
                    
                # events to words
                tp_words, ys = self.events_to_words(tp_events, task)
                
                # slice to chunks so that max length = max_len (default: 512)
                if segment_mode == 'token':
                    slice_words, slice_ys, last_word_idx = self.slice_words_and_ys(tp_words, ys, path, max_len, task)
                    last_word_idxs = None
                    last_segment_bar = None
                    segment_bar = None
    
                    bar_start_token_idxs = self.get_bar_start_token_idxs(tp_words)
                    
                    segment_piece_dir = Path(out_dir) / 'pkls'
                    segment_piece_dir.mkdir(parents=True, exist_ok=True)
                    out_piece_file = segment_piece_dir / (Path(path).stem + '.pkl')
                    out_dict = {
                        'words': tp_words,
                        'sliced_words': slice_words,
                        'bar_start_token_idxs': bar_start_token_idxs,
                    }
                    with open(out_piece_file, 'wb') as f:
                        pickle.dump(out_dict, f)
                    
                    
                ### TODO: n (2, 4, 8) bars segmentation (Bar level sliced)
                # accum_segment_idx = len(all_words)
                elif segment_mode == 'bar':
                    last_word_idx = None
                    slice_words, slice_ys, last_word_idxs, last_segment_bar, segment_bar = self.slice_words_and_ys_bar(tp_words, ys, path, max_len, n_seg_bar, task, len(all_words))
                    
                self.update_piece_segment_info(all_piece_info, tp_events, segmented_events, all_words, slice_words, path, max_len, last_word_idx, last_word_idxs, last_segment_bar, tp_val, segment_bar)
                assert len(all_piece_info[path]['bar_start_times']) == len(bar_start_token_idxs)
                all_words = all_words + slice_words
                all_ys = all_ys + slice_ys

            
            
        return all_words, all_ys, all_piece_info, segmented_events

    def events2midi(self, events, output_path, prompt_path=None, tempo_changes=None):
        """
            Given melody events, convert back to midi
        """
        temp_notes, temp_tempos = [], []

        for i, event in enumerate(events):
            if len(event) == 1:         # [Bar]
                temp_notes.append('Bar')
                temp_tempos.append('Bar')

            elif len(event) == 5:       # [Bar, Position, Pitch, Duration, Velocity]
                # start time and end time from position
                position = int(event[1].value.split('/')[0]) - 1
                # pitch
                pitch = int(event[2].value)
                # duration
                index = int(event[3].value)
                duration = DEFAULT_DURATION_BINS[index]
                # velocity
                index = int(event[4].value)
                velocity = int(DEFAULT_VELOCITY_BINS[index])
                # adding
                temp_notes.append([position, velocity, pitch, duration])

            elif len(event) == 4:  # [Bar, Position, Pitch, Duration]
                if i != 0 and event[0].value=='New':
                    temp_notes.append('Bar')
                # start time and end time from position
                position = int(event[1].value.split('/')[0]) - 1
                # pitch
                pitch = int(event[2].value)
                # duration
                index = int(event[3].value)
                duration = DEFAULT_DURATION_BINS[index]
                # adding
                VEL = 80
                temp_notes.append([position, VEL, pitch, duration])
            else:                       # [Position, Tempo Class, Tempo Value]
                position = int(event[0].value.split('/')[0]) - 1
                if event[1].value == 'slow':
                    tempo = DEFAULT_TEMPO_INTERVALS[0].start + int(event[2].value)
                elif event[1].value == 'mid':
                    tempo = DEFAULT_TEMPO_INTERVALS[1].start + int(event[2].value)
                elif event[1].value == 'fast':
                    tempo = DEFAULT_TEMPO_INTERVALS[2].start + int(event[2].value)
                temp_tempos.append([position, tempo])

        # get specific time for notes
        ticks_per_beat = DEFAULT_RESOLUTION
        ticks_per_bar = DEFAULT_RESOLUTION * 4 # assume 4/4
        notes = []
        current_bar = 0
        for note in temp_notes:
            if note == 'Bar':
                current_bar += 1
            else:
                position, velocity, pitch, duration = note
                # position (start time)
                current_bar_st = current_bar * ticks_per_bar
                current_bar_et = (current_bar + 1) * ticks_per_bar
                flags = np.linspace(current_bar_st, current_bar_et, DEFAULT_FRACTION, endpoint=False, dtype=int)
                st = flags[position]
                # duration (end time)
                et = st + duration
                notes.append(miditoolkit.Note(velocity, pitch, st, et))

        # get specific time for tempos
        tempos = []
        current_bar = 0
        for tempo in temp_tempos:
            if tempo == 'Bar':
                current_bar += 1
            else:
                position, value = tempo
                # position (start time)
                current_bar_st = current_bar * ticks_per_bar
                current_bar_et = (current_bar + 1) * ticks_per_bar
                flags = np.linspace(current_bar_st, current_bar_et, DEFAULT_FRACTION, endpoint=False, dtype=int)
                st = flags[position]
                tempos.append([int(st), value])
        # write
        if prompt_path:
            midi = miditoolkit.midi.parser.MidiFile(prompt_path)
            #
            last_time = DEFAULT_RESOLUTION * 4 * 4
            # note shift
            for note in notes:
                note.start += last_time
                note.end += last_time
            midi.instruments[0].notes.extend(notes)
            # tempo changes
            temp_tempos = []
            for tempo in midi.tempo_changes:
                if tempo.time < DEFAULT_RESOLUTION*4*4:
                    temp_tempos.append(tempo)
                else:
                    break
            for st, bpm in tempos:
                st += last_time
                temp_tempos.append(miditoolkit.midi.containers.TempoChange(bpm, st))
            midi.tempo_changes = temp_tempos
        else:
            midi = miditoolkit.midi.parser.MidiFile()
            midi.ticks_per_beat = DEFAULT_RESOLUTION
            # write instrument
            inst = miditoolkit.midi.containers.Instrument(0, is_drum=False)
            inst.notes = notes
            midi.instruments.append(inst)
            # write tempo
            if tempo_changes==None:
                tempo_changes = []
                for st, bpm in tempos:
                    tempo_changes.append(miditoolkit.midi.containers.TempoChange(bpm, st))
            
            midi.tempo_changes = tempo_changes
        
        # write  
        midi.dump(output_path)
        # print(f"midi file is saved at {output_path}")

        return


    def get_tempo_events(self, seg_idx, seg_info, start_tick, end_tick, do_transpose=False):
        # based on seg_idx, get the original midi idx
        midi_file = None
        transposed_value = 0
        if do_transpose:
            # print('do transpose')
            transpose_vals = TRANSPOSE_VALS
        else:
            transpose_vals = [0]
        
        for tr_val in transpose_vals:
            for key, value in seg_info.items():
                if value[tr_val]['start slice idx'] <= seg_idx <= value[tr_val]['end slice idx']:
                    midi_file = key
                    transposed_value = tr_val
                    break
                
        midi_obj = miditoolkit.midi.parser.MidiFile(midi_file)
        selected_tempo_events = []
        # here, appropriate tempo changes events between start tick ~ end tick are selected and shifted according to the segment bar  
        seg_start_bar = seg_info[midi_file][transposed_value]['segment bar'][seg_idx]['start bar']
        seg_start_bar_tick = seg_info[midi_file]['bar_start_times'][seg_start_bar]
        for tempo_event in midi_obj.tempo_changes:
            if start_tick <= tempo_event.time <= end_tick:
                shifted_tempo_event = copy.deepcopy(tempo_event)
                shifted_tempo_event.time -= seg_start_bar_tick # shift
                selected_tempo_events.append(shifted_tempo_event)
        
        return selected_tempo_events
    
    def token_to_event_str(self, input_midi):
        # input midi is tokenzied sequence tensor
        decoded_midi = []
        for l in range(len(input_midi)): # tgt_len * n_vocab
            decoded_midi.append([self.word2event[etype][input_midi[l][i].item()] 
                for i, etype in enumerate(self.word2event)])
    
        return decoded_midi
    
    def event_str_to_events(self, decoded_midi):
        """
        event_string [['Bar New', 'Position 2/16', 'Pitch 58', 'Duration 3'], ['Bar Continue', 'Position 2/16', 'Pitch 57', 'Duration 1'], ...]
        to event [[Event('Bar', 'New'), Event('Position', '2/16'), ...]]
        """
        events = []
        
        for note_list in decoded_midi:
            note_events = []
            
            for item in note_list:
                # Split each item into components
                components = item.split()
                
                # Skip the event if the value is '<PAD>'
                if components[1] == "<PAD>":
                    continue
                
                if components[0] == "Bar":
                    name = "Bar"
                    value = components[1]
                    text = ""
                    event_type = -1
                    time = None
                
                elif components[0] == "Position":
                    name = "Position"
                    value = components[1]
                    text = components[1]
                    event_type = -1
                    time = None  

                elif components[0] == "Pitch":
                    name = "Pitch"
                    value = int(components[1])
                    text = components[1]
                    event_type = -1
                    time = None

                elif components[0] == "Duration":
                    name = "Duration"
                    value = int(components[1])
                    text = components[1]
                    event_type = -1
                    time = None

                # Create an Event object
                event = Event(name=name, time=time, value=value, text=text, Type=event_type)
                note_events.append(event)
            
            if note_events:  # Only add non-empty note events
                events.append(note_events)
        
        return events