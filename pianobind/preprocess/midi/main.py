import os
from pathlib import Path
import glob
import pickle
import argparse
import numpy as np
from .model import *
from .midi_synth_dir import get_median_tempo, save_wav_from_midi
from tqdm import tqdm
import json
import yaml
import argparse
    
def get_args():
    parser = argparse.ArgumentParser(description='')
    
    # Path to the configuration file
    parser.add_argument('--prepare_data_config', type=str, default='./pianobind/preprocess/midi/midibert_utils/prepare_data_config.yaml', help='Path to the config file')
    
    args = parser.parse_args()
    
    # Load the yaml configuration file
    with open(args.prepare_data_config, 'r') as file:
        config = yaml.safe_load(file)
        
    # Override default parser arguments with config values
    for key, value in config.items():
        if isinstance(value, list):
            parser.add_argument(f'--{key}', type=type(value[0]), nargs='+', default=value)
        else:
            parser.add_argument(f'--{key}', type=type(value), default=value)
    
    args = parser.parse_args()
        
    if args.task == 'melody' and args.dataset != 'pop909':
        print('[error] melody task is only supported for pop909 dataset')
        exit(1)
    elif args.task == 'composer' and args.dataset != 'pianist8':
        print('[error] composer task is only supported for pianist8 dataset')
        exit(1)
    elif args.task == 'emotion' and args.dataset != 'emopia':
        print('[error] emotion task is only supported for emopia dataset')
        exit(1)
    elif args.dataset == None and args.input_dir == None and args.input_file == None:
        print('[error] Please specify the input directory or dataset')
        exit(1)

    return args


def extract(files, args, model, mode=''):
    '''
    files: list of midi path
    mode: 'train', 'valid', 'test', ''
    args.input_dir: '' or the directory to your custom data
    args.output_dir: the directory to store the data (and answer data) in CP representation
    '''
    
    assert len(files)

    print(f'Number of {mode} files: {len(files)}') 

    segments, ans, seg_info, segmented_events = model.prepare_data(files, args.output_dir, args.task, args.segment_mode, args.do_transpose, int(args.max_len), int(args.n_seg_bar))

    dataset = args.dataset if args.dataset != 'pianist8' else 'composer'

    if args.segment_mode == 'token':
        ext = 'npy'
    elif args.segment_mode == 'bar':
        ext = 'pkl'
    else:
        NotImplementedError()
    
    if args.input_dir != '' or args.input_file != '':
        name = args.input_dir or args.input_file
        if args.name == '':
            args.name = Path(name).stem
            #### TODO: 
            # args.name = 'corpus'
        output_file = os.path.join(args.output_dir, f'{args.name}.{ext}')
    elif dataset == 'composer' or dataset == 'emopia' or dataset == 'pop909':
        output_file = os.path.join(args.output_dir, f'{dataset}_{mode}.{ext}')
    elif dataset == 'pop1k7' or dataset == 'ASAP':
        output_file = os.path.join(args.output_dir, f'{dataset}.{ext}')
        
    if args.segment_mode == 'token':
        segments = np.array(segments)
        np.save(output_file, segments)
        print(f'Data shape: {segments.shape}, saved at {output_file}')
    elif args.segment_mode == 'bar':
        with open(output_file, 'wb') as f:   
            pickle.dump(segments, f)
        print(f'Data shape: {len(segments)}, saved at {output_file}')
    else:
        NotImplementedError()
            
    
    # save segment info dictionary as json
    with open(output_file.replace(f'.{ext}', '_segment_info.pkl'), 'wb') as f:
        pickle.dump(seg_info, f)
        
    # also save dictionary as json
    with open(output_file.replace(f'.{ext}', '_segment_info.json'), "w") as outfile:
        json.dump(seg_info, outfile, indent=4, sort_keys=False)

    
    # save segment midi and wav
    if args.save_segments:
        mid_dir = os.path.join(args.output_dir, 'seg_midi')
        os.makedirs(mid_dir, exist_ok=True)
        for seg, seg_events in tqdm(segmented_events.items()):
            if os.path.exists(f'{mid_dir}/{seg}_rendered.wav'):
                continue
            if len(seg_events) < 5:
                print(seg)
                continue
            # start and end tick is absolute tick that are recorded in the event data class
            start_tick=seg_events[0][1].time
            end_tick=seg_events[-1][1].time
            
            if args.not_use_tempo_changes:  
                tempo_changes = None      
            else:
                tempo_changes = model.get_tempo_events(seg, seg_info, start_tick, end_tick, do_transpose=args.do_transpose) # here, appropriate tempo changes events between start tick ~ end tick are selected and shifted according to the segment bar  
            
            model.events2midi(seg_events, f'{mid_dir}/{seg}.mid', tempo_changes=tempo_changes)
            
            if args.save_segments_audio:
                if args.not_use_tempo_changes:
                    mid_tempo = 120
                else:
                    mid_tempo = get_median_tempo(tempo_changes)

                save_wav_from_midi(f'{mid_dir}/{seg}.mid', f'{mid_dir}/{seg}_rendered.wav', qpm=mid_tempo)

    if args.task != '':
        if args.task == 'melody' or args.task == 'velocity':
            ans_file = os.path.join(args.output_dir, f'{dataset}_{mode}_{args.task[:3]}ans.{ext}')
        elif args.task == 'composer' or args.task == 'emotion':
            ans_file = os.path.join(args.output_dir, f'{dataset}_{mode}_ans.{ext}')
        
        if args.segment_mode == 'token':
            ans = np.array(ans)
            np.save(ans_file, ans)
            print(f'Answer shape: {ans.shape}, saved at {ans_file}')
    
        elif args.segment_mode == 'bar':
            with open(ans_file, 'wb') as f:   
                pickle.dump(ans, f)
            print(f'Data shape: {len(ans)}, saved at {output_file}')
        else:
            NotImplementedError()

def main(): 
    args = get_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Convert args to a dictionary
    args_dict = vars(args)

    # Save to YAML file
    with open(Path(args.output_dir) / 'args.yaml', 'w') as file:
        yaml.dump(args_dict, file, default_flow_style=False)
    
    # initialize model
    model = CP(dict=args.dict)

    if args.dataset == 'pop909':
        dataset = 'pop909_processed'
    elif args.dataset == 'emopia':
        dataset = 'EMOPIA_1.0'
    elif args.dataset == 'pianist8':
        dataset = 'joann8512-Pianist8-ab9f541'

    if args.dataset == 'pop909' or args.dataset == 'emopia':
        train_files = glob.glob(f'../../Data/Dataset/{dataset}/train/*.mid')
        valid_files = glob.glob(f'../../Data/Dataset/{dataset}/valid/*.mid')
        test_files = glob.glob(f'../../Data/Dataset/{dataset}/test/*.mid')

    elif args.dataset == 'pianist8':
        train_files = glob.glob(f'../../Data/Dataset/{dataset}/train/*/*.mid')
        valid_files = glob.glob(f'../../Data/Dataset/{dataset}/valid/*/*.mid')
        test_files = glob.glob(f'../../Data/Dataset/{dataset}/test/*/*.mid')

    elif args.dataset == 'pop1k7':
        files = list(glob.glob(f'./Data/Dataset/midi_{args.pop1k7_mode}/*/*.midi'))
        files += list(glob.glob(f'./Data/Dataset/midi_{args.pop1k7_mode}/*/*.mid'))
        
    elif args.dataset == 'ASAP':
        files = pickle.load(open('../../Data/Dataset/ASAP_song.pkl', 'rb'))
        files = [f'../../Dataset/asap-dataset/{file}' for file in files]

    elif args.input_dir:
        # files = list(glob.glob(f'{args.input_dir}/*.mid')) + list(glob.glob(f'{args.input_dir}/*.midi'))
        # files += (list(glob.glob(f'{args.input_dir}/*/*.mid')) + list(glob.glob(f'{args.input_dir}/*/*.midi')))
        # from the input directory, recursively search for all midi files
        files = []
        for root, dirs, _files in tqdm(os.walk(args.input_dir)):
            for file in _files:
                if file.endswith('.mid') or file.endswith('.midi'):
                    files.append(os.path.join(root, file))


    elif args.input_file:
        files = [args.input_file]

    else:
        print('not supported')
        exit(1)

    files = sorted(files)
    
    if args.debug:
        files = list(files)[:3]

    if args.dataset in {'pop909', 'emopia', 'pianist8'}:
        extract(train_files, args, model, 'train')
        extract(valid_files, args, model, 'valid')
        extract(test_files, args, model, 'test')
    else:
        # in one single file
        extract(files, args, model)
        

if __name__ == '__main__':
    main()
