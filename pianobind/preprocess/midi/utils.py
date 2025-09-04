import numpy as np
import miditoolkit
import copy

# parameters for input
DEFAULT_VELOCITY_BINS = np.array([ 0, 32, 48, 64, 80, 96, 128])     # np.linspace(0, 128, 32+1, dtype=np.int)
DEFAULT_FRACTION = 16
DEFAULT_DURATION_BINS = np.arange(60, 3841, 60, dtype=int)
DEFAULT_TEMPO_INTERVALS = [range(30, 90), range(90, 150), range(150, 210)]

# parameters for output
DEFAULT_RESOLUTION = 480

PITCH_MIN = 22
PITCH_MAX = 107

# define "Item" for general storage
class Item(object):
    def __init__(self, name, start, end, velocity, pitch, Type):
        self.name = name
        self.start = start
        self.end = end
        self.velocity = velocity
        self.pitch = pitch
        self.Type = Type

    def __repr__(self):
        return 'Item(name={}, start={}, end={}, velocity={}, pitch={}, Type={})'.format(
            self.name, self.start, self.end, self.velocity, self.pitch, self.Type)

# read notes and tempo changes from midi (assume there is only one track)
def read_items(file_path, num_of_instr=None):
    midi_obj = miditoolkit.midi.parser.MidiFile(file_path)
    # note
    note_items = []
    if num_of_instr is None:
        num_of_instr = len(midi_obj.instruments)
    else:
        print(f'use instrument {num_of_instr}')
    
    for i in range(num_of_instr):
        notes = midi_obj.instruments[i].notes
        notes.sort(key=lambda x: (x.start, x.pitch))

        for note in notes:
            if note.pitch >= PITCH_MIN and note.pitch <= PITCH_MAX: # ignore invalid pitches from transcription data
                note_items.append(Item(
                    name='Note',
                    start=note.start, 
                    end=note.end, 
                    velocity=note.velocity, 
                    pitch=note.pitch,
                    Type=i))
                
    note_items.sort(key=lambda x: x.start)
    
    # tempo
    tempo_items = []
    for tempo in midi_obj.tempo_changes:
        tempo_items.append(Item(
            name='Tempo',
            start=tempo.time,
            end=None,
            velocity=None,
            pitch=int(tempo.tempo),
            Type=-1))
    tempo_items.sort(key=lambda x: x.start)

    # expand to all beat
    max_tick = tempo_items[-1].start
    existing_ticks = {item.start: item.pitch for item in tempo_items}
    wanted_ticks = np.arange(0, max_tick+1, DEFAULT_RESOLUTION)
    output = []
    for tick in wanted_ticks:
        if tick in existing_ticks:
            output.append(Item(
                name='Tempo',
                start=tick,
                end=None,
                velocity=None,
                pitch=existing_ticks[tick],
                Type=-1))
        else:
            output.append(Item(
                name='Tempo',
                start=tick,
                end=None,
                velocity=None,
                pitch=output[-1].pitch,
                Type=-1))
    tempo_items = output
    return note_items, tempo_items


class Event(object):
    def __init__(self, name, time, value, text, Type):
        self.name = name
        self.time = time
        self.value = value
        self.text = text
        self.Type = Type

    def __repr__(self):
        return 'Event(name={}, time={}, value={}, text={}, Type={})'.format(
            self.name, self.time, self.value, self.text, self.Type)


def item2event(groups, task):
    events = []
    n_downbeat = 0
    for i in range(len(groups)):
        if 'Note' not in [item.name for item in groups[i][1:-1]]:
            continue
        bar_st, bar_et = groups[i][0], groups[i][-1]
        n_downbeat += 1
        new_bar = True
        
        for item in groups[i][1:-1]:
            if item.name != 'Note':
                continue
            note_tuple = []

            # Bar
            if new_bar:
                BarValue = 'New' 
                new_bar = False
            else:
                BarValue = "Continue"
            note_tuple.append(Event(
                name='Bar',
                time=None, 
                value=BarValue,
                text='{}'.format(n_downbeat),
                Type=-1))

            # Position
            flags = np.linspace(bar_st, bar_et, DEFAULT_FRACTION, endpoint=False)
            index = np.argmin(abs(flags-item.start))
            note_tuple.append(Event(
                name='Position', 
                time=item.start,
                value='{}/{}'.format(index+1, DEFAULT_FRACTION),
                text='{}'.format(item.start),
                Type=-1))
            
            # Pitch
            velocity_index = np.searchsorted(DEFAULT_VELOCITY_BINS, item.velocity, side='right') - 1

            if task == 'melody':
                pitchType = item.Type
            elif task == 'velocity':
                pitchType = velocity_index
            else:
                pitchType = -1
                
            note_tuple.append(Event(
                name='Pitch',
                time=item.start, 
                value=item.pitch,
                text='{}'.format(item.pitch),
                Type=pitchType))

            # Duration
            duration = item.end - item.start
            index = np.argmin(abs(DEFAULT_DURATION_BINS-duration))
            note_tuple.append(Event(
                name='Duration',
                time=item.start,
                value=index,
                text='{}/{}'.format(duration, DEFAULT_DURATION_BINS[index]),
                Type=-1))

            events.append(note_tuple)

    return events


def quantize_items(items, ticks=120):
    grids = np.arange(0, items[-1].start, ticks, dtype=int)
    # process
    for item in items:
        index = np.argmin(abs(grids - item.start))
        shift = grids[index] - item.start
        item.start += shift
        item.end += shift
    return items      


def group_items(items, max_time, ticks_per_bar=DEFAULT_RESOLUTION*4):
    items.sort(key=lambda x: x.start)
    downbeats = np.arange(0, max_time+ticks_per_bar, ticks_per_bar)
    groups = []
    for db1, db2 in zip(downbeats[:-1], downbeats[1:]):
        insiders = []
        for item in items:
            if (item.start >= db1) and (item.start < db2):
                insiders.append(item)
        overall = [db1] + insiders + [db2]
        groups.append(overall)
    return groups

def get_ticks_to_seconds_grid(midi_obj) -> np.array:
    # max_tick = max([note.end for inst in midi_obj.instruments for note in inst.notes])
    # max_tick = max(max_tick, max([note.start for inst in midi_obj.instruments for note in inst.notes]))
    # max_tick = max(max_tick, midi_obj.max_tick)
    
    ticks_to_seconds = np.zeros(midi_obj.max_tick + 1)
    ticks_per_beat = midi_obj.ticks_per_beat
    tempo_changes = midi_obj.tempo_changes  # Assumed to be a list of (tick, bpm) tuples

    # Initialize time tracking variables
    accumulated_time_seconds = 0.0
    current_tempo_index = 0
    current_tempo_change = tempo_changes[current_tempo_index]
    current_bpm = current_tempo_change.tempo
    seconds_per_beat = 60.0 / current_bpm
    ticks_per_second = ticks_per_beat / seconds_per_beat
    
    # Iterate through each tick up to the maximum tick value
    final_tick = midi_obj.max_tick

    for tick in range(final_tick + 1):
        # Update tempo if we reach a new tempo change tick
        if current_tempo_index < len(tempo_changes) - 1 and tick == tempo_changes[current_tempo_index + 1].time:
            current_tempo_index += 1
            current_tempo_change = tempo_changes[current_tempo_index]
            current_bpm = current_tempo_change.tempo
            seconds_per_beat = 60.0 / current_bpm
            ticks_per_second = ticks_per_beat / seconds_per_beat

        # Store accumulated time for the current tick
        ticks_to_seconds[tick] = accumulated_time_seconds

        # Advance accumulated time by 1 tick in seconds
        accumulated_time_seconds += 1 / ticks_per_second

    return ticks_to_seconds

# Function to find the closest bar start time for a given start_sec
def find_closest_bar(bar_start_times_sec, start_sec):
    bar_times = np.array(list(bar_start_times_sec.values()))
    closest_idx = np.argmin(np.abs(bar_times - start_sec))
    closest_bar = list(bar_start_times_sec.keys())[closest_idx]
    closest_time = bar_start_times_sec[closest_bar]
    return int(closest_bar), closest_time