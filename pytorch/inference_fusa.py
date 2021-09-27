import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../utils'))
import numpy as np
import argparse
import librosa
import matplotlib.pyplot as plt
import torch
import pandas as pd

from utilities import create_folder, get_filename
from models import *
from pytorch_utils import move_data_to_device
import config

def get_label_transforms(dataset_name):
    taxonomy_path = "fusa_taxonomy.json"
    a = pd.read_json(taxonomy_path).T[dataset_name].to_dict()
    transforms = {}
    for key, values in a.items():
        for value in values:
            transforms[value] = key
    return transforms

def audio_tagging(args):
    """Inference audio tagging result of an audio clip.
    """

    # Arugments & parameters
    sample_rate = args.sample_rate
    window_size = args.window_size
    hop_size = args.hop_size
    mel_bins = args.mel_bins
    fmin = args.fmin
    fmax = args.fmax
    model_type = args.model_type
    checkpoint_path = args.checkpoint_path
    audios_path = args.audios_path
    meta_path = args.meta_path
    dataset_name = args.dataset_name
    device = torch.device('cuda') if args.cuda and torch.cuda.is_available() else torch.device('cpu')
    
    classes_num = config.classes_num
    labels = config.labels

    # Model
    Model = eval(model_type)
    model = Model(sample_rate=sample_rate, window_size=window_size, 
        hop_size=hop_size, mel_bins=mel_bins, fmin=fmin, fmax=fmax, 
        classes_num=classes_num)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])

    # Parallel
    if 'cuda' in str(device):
        model.to(device)
        print('GPU number: {}'.format(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model)
    else:
        print('Using CPU.')
    
    # Load audio
    df = pd.read_csv(meta_path)

    output_df = pd.DataFrame([],
        columns=['audio_name', 'dataset', 'label', 'output_1', 'output_2', 'output_3', 'fusa_output_1', 'fusa_output_2', 'fusa_output_3', 'acc_1', 'acc_2', 'acc_3'])

    audioset_transforms = get_label_transforms("AudioSet")

    number_files = len(os.listdir(audios_path))

    for i, audio in enumerate(os.listdir(audios_path)):
        audio_path = os.path.join(audios_path, audio)
        (waveform, _) = librosa.core.load(audio_path, sr=sample_rate, mono=True)

        waveform = waveform[None, :]    # (1, audio_length)
        try:
            waveform = move_data_to_device(waveform, device)

            # Forward
            with torch.no_grad():
                model.eval()
                batch_output_dict = model(waveform, None)

            clipwise_output = batch_output_dict['clipwise_output'].data.cpu().numpy()[0]
            """(classes_num,)"""

            sorted_indexes = np.argsort(clipwise_output)[::-1]

            output_1 = np.array(labels)[sorted_indexes[0]]
            output_2 = np.array(labels)[sorted_indexes[1]]
            output_3 = np.array(labels)[sorted_indexes[2]]

            if dataset_name == "ESC":
                mask = df.filename == audio
                target = df["category"].loc[mask].item()

            elif dataset_name == "UrbanSound":
                mask = df.slice_file_name == audio
                target = df["class"].loc[mask].item()

            if output_1 in audioset_transforms:
                output_fusa_1 = audioset_transforms[output_1]
            else:
                output_fusa_1 = ""

            if output_2 in audioset_transforms:
                output_fusa_2 = audioset_transforms[output_2]
            else:
                output_fusa_2 = ""

            if output_3 in audioset_transforms:
                output_fusa_3 = audioset_transforms[output_3]
            else:
                output_fusa_3 = ""

            acc_1 = clipwise_output[sorted_indexes[0]]
            acc_2 = clipwise_output[sorted_indexes[1]]
            acc_3 = clipwise_output[sorted_indexes[2]]

            #print(audio, dataset_name, target, output, output_fusa, round(acc, 3))
            output_df = output_df.append(
                {
                    'audio_name' : audio ,
                    'dataset' : dataset_name,
                    'label' : target,
                    'output_1' : output_1 ,
                    'output_2' : output_2 ,
                    'output_3' : output_3 ,
                    'fusa_output_1' : output_fusa_1,
                    'fusa_output_2' : output_fusa_2,
                    'fusa_output_3' : output_fusa_3,
                    'acc_1' : round(acc_1, 3),
                    'acc_2' : round(acc_2, 3),
                    'acc_3' : round(acc_3, 3),
                } ,
                ignore_index=True)
        except Exception as e:
            output_df = output_df.append(
                {
                    'audio_name' : audio ,
                    'dataset' : dataset_name,
                    'label' : '',
                    'output_1' : '',
                    'output_2' : '',
                    'output_3' : '',
                    'fusa_output_1' : '',
                    'fusa_output_2' : '',
                    'fusa_output_3' : '',
                    'acc_1' : '',
                    'acc_2' : '',
                    'acc_3' : '',
                } ,
                ignore_index=True)
        
        print(f"{i}/{number_files}")    
    output_df.to_csv(f'fusa_{dataset_name}_results_v5.csv')

def sound_event_detection(args):
    """Inference sound event detection result of an audio clip.
    """

    # Arugments & parameters
    sample_rate = args.sample_rate
    window_size = args.window_size
    hop_size = args.hop_size
    mel_bins = args.mel_bins
    fmin = args.fmin
    fmax = args.fmax
    model_type = args.model_type
    checkpoint_path = args.checkpoint_path
    audios_path = args.audios_path
    meta_path = args.meta_path
    dataset_name = args.dataset_name
    device = torch.device('cuda') if args.cuda and torch.cuda.is_available() else torch.device('cpu')

    classes_num = config.classes_num
    labels = config.labels
    frames_per_second = sample_rate // hop_size

    # Paths
    fig_path = os.path.join('results', '{}.png'.format(get_filename(audio_path)))
    create_folder(os.path.dirname(fig_path))

    # Model
    Model = eval(model_type)
    model = Model(sample_rate=sample_rate, window_size=window_size, 
        hop_size=hop_size, mel_bins=mel_bins, fmin=fmin, fmax=fmax, 
        classes_num=classes_num)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])

    # Parallel
    print('GPU number: {}'.format(torch.cuda.device_count()))
    model = torch.nn.DataParallel(model)

    if 'cuda' in str(device):
        model.to(device)
    
    # Load audio
    (waveform, _) = librosa.core.load(audio_path, sr=sample_rate, mono=True)

    waveform = waveform[None, :]    # (1, audio_length)
    waveform = move_data_to_device(waveform, device)

    # Forward
    with torch.no_grad():
        model.eval()
        batch_output_dict = model(waveform, None)

    framewise_output = batch_output_dict['framewise_output'].data.cpu().numpy()[0]
    """(time_steps, classes_num)"""

    print('Sound event detection result (time_steps x classes_num): {}'.format(
        framewise_output.shape))

    sorted_indexes = np.argsort(np.max(framewise_output, axis=0))[::-1]

    top_k = 10  # Show top results
    top_result_mat = framewise_output[:, sorted_indexes[0 : top_k]]    
    """(time_steps, top_k)"""

    # Plot result    
    stft = librosa.core.stft(y=waveform[0].data.cpu().numpy(), n_fft=window_size, 
        hop_length=hop_size, window='hann', center=True)
    frames_num = stft.shape[-1]

    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(10, 4))
    axs[0].matshow(np.log(np.abs(stft)), origin='lower', aspect='auto', cmap='jet')
    axs[0].set_ylabel('Frequency bins')
    axs[0].set_title('Log spectrogram')
    axs[1].matshow(top_result_mat.T, origin='upper', aspect='auto', cmap='jet', vmin=0, vmax=1)
    axs[1].xaxis.set_ticks(np.arange(0, frames_num, frames_per_second))
    axs[1].xaxis.set_ticklabels(np.arange(0, frames_num / frames_per_second))
    axs[1].yaxis.set_ticks(np.arange(0, top_k))
    axs[1].yaxis.set_ticklabels(np.array(labels)[sorted_indexes[0 : top_k]])
    axs[1].yaxis.grid(color='k', linestyle='solid', linewidth=0.3, alpha=0.3)
    axs[1].set_xlabel('Seconds')
    axs[1].xaxis.set_ticks_position('bottom')

    plt.tight_layout()
    plt.savefig(fig_path)
    print('Save sound event detection visualization to {}'.format(fig_path))

    return framewise_output, labels


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Example of parser. ')
    subparsers = parser.add_subparsers(dest='mode')

    parser_at = subparsers.add_parser('audio_tagging')
    parser_at.add_argument('--sample_rate', type=int, default=32000)
    parser_at.add_argument('--window_size', type=int, default=1024)
    parser_at.add_argument('--hop_size', type=int, default=320)
    parser_at.add_argument('--mel_bins', type=int, default=64)
    parser_at.add_argument('--fmin', type=int, default=50)
    parser_at.add_argument('--fmax', type=int, default=14000) 
    parser_at.add_argument('--model_type', type=str, required=True)
    parser_at.add_argument('--checkpoint_path', type=str, required=True)
    parser_at.add_argument('--audios_path', type=str, required=True) 
    parser_at.add_argument('--meta_path', type=str, required=True) 
    parser_at.add_argument('--dataset_name', type=str, required=True) 
    parser_at.add_argument('--cuda', action='store_true', default=False)

    parser_sed = subparsers.add_parser('sound_event_detection')
    parser_sed.add_argument('--sample_rate', type=int, default=32000)
    parser_sed.add_argument('--window_size', type=int, default=1024)
    parser_sed.add_argument('--hop_size', type=int, default=320)
    parser_sed.add_argument('--mel_bins', type=int, default=64)
    parser_sed.add_argument('--fmin', type=int, default=50)
    parser_sed.add_argument('--fmax', type=int, default=14000) 
    parser_sed.add_argument('--model_type', type=str, required=True)
    parser_sed.add_argument('--checkpoint_path', type=str, required=True)
    parser_sed.add_argument('--audios_path', type=str, required=True)
    parser_sed.add_argument('--meta_path', type=str, required=True) 
    parser_sed.add_argument('--dataset_name', type=str, required=True) 
    parser_sed.add_argument('--cuda', action='store_true', default=False)
    
    args = parser.parse_args()

    if args.mode == 'audio_tagging':
        audio_tagging(args)

    elif args.mode == 'sound_event_detection':
        sound_event_detection(args)

    else:
        raise Exception('Error argument!')