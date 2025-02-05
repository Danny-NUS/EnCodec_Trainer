import torchaudio
# import pesto
import multiprocessing
import soundfile as sf
import pyworld as pw
# You can also predict pitches from audio files directly
import os
import numpy as np
import math
from scipy.signal import resample_poly
import torchaudio
import torchaudio.transforms as T

        
def extractor_pyworld(wavlist):
    cnt = 0
    for wavpath in wavlist:
        if cnt and cnt % 1000 == 0:
            print('processing:', cnt, wavpath)

        x, outFs = sf.read(wavpath)
        sr = 24000
        x = resample_poly(x, sr, outFs)[:48000]
        import pdb
        pdb.set_trace()


        for frame_period in frame_periods:
            try:

                f0, t = pw.dio(x, sr, f0_floor=f0_floor, f0_ceil=f0_ceil, frame_period=frame_period)
                f0 = pw.stonemask(x, f0, t, sr)
                print(x.size, x)
            except Exception as e:
                print(e)
                print(wavpath)
                continue

            uv = np.zeros(f0.shape).astype('float32')
            uv[np.where(f0 > 0)] = 1

            f0path = wavpath[:-4] + '.pw.f0.' + str(frame_period) + 'ms.npy'
            uvpath = wavpath[:-4] + '.pw.uv.' + str(frame_period) + 'ms.npy'

            non_zero_values = f0[f0 != 0]  # Extract non-zero values
            normalized_values = (non_zero_values - non_zero_values.min()) / (non_zero_values.max() - non_zero_values.min())

            scaled_values = np.round(normalized_values * 254).astype(int) + 1

            f0[f0 != 0] = scaled_values
            f0 = f0.astype(int)

            import pdb
            pdb.set_trace()
            
            np.save(f0path, f0)
            np.save(uvpath, uv)

            

        cnt += 1
    # return f0, uv

def pitch_to_f0(pitch):
    if pitch == 0:
        return 0
    return 27.5 * math.pow(2, (pitch - 21) / 12)
    
def f0ToPitch(f0):
    return np.log2(f0 / 27.5) * 12 + 21

def coarse_f0(f0, f0_bin):
    f0_mel = 1127 * np.log(1 + f0 / 700)
    f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * (
        f0_bin - 2
    ) / (f0_mel_max - f0_mel_min) + 1

    # use 0 or 1
    f0_mel[f0_mel <= 1] = 1
    f0_mel[f0_mel > f0_bin - 1] = f0_bin - 1
    f0_coarse = np.rint(f0_mel).astype(int)
    assert f0_coarse.max() <= (f0_bin - 1) and f0_coarse.min() >= 1, (
        f0_coarse.max(),
        f0_coarse.min(),
    )
    return f0_coarse

f0_floor = 60
f0_ceil = 1400
frame_periods = [5/3]
f0_mel_min = 1127 * np.log(1 + f0_floor / 700)
f0_mel_max = 1127 * np.log(1 + f0_ceil / 700)
# f0_bins = [128,256,512]

wavpath = ['/home/junchuan/EnCodec_Trainer/audio/output_000002.wav']
extractor_pyworld(wavpath)

import pdb
pdb.set_trace()
# pesto_pitch('BAC009S0219W0272.f0.npz')
    
# 1. modify dataset:
# dataset = 'aishell-1'
# rootdir = '/data2/xintong/aishell/data_aishell/wav/train'
# f0 file: BAC009S0764W0346.pw.f0.npy

# dataset = 'latic'
# rootdir = '/data2/xintong/LATIC/WAVE'

# 2. frame_preiods and f0_bins
    
# dataset = 'aishell-1'
# rootdir = '/data2/xintong/aishell/data_aishell/wav/test'

# f0_floor = 60
# f0_ceil = 1400
# frame_periods = [5,20,40]
# f0_mel_min = 1127 * np.log(1 + f0_floor / 700)
# f0_mel_max = 1127 * np.log(1 + f0_ceil / 700)
# # f0_bins = [128,256,512]
# f0_bins = [256]
# file_list = [[],[],[],[],[],[],[],[]]
# # print(file_list)
# for root, dirs, files in os.walk(rootdir):
#     for file in files:
#         # latic
#         if dataset == 'latic':
#             if '.WAV' not in file:
#                 continue
#             filepath = os.path.join(root, file)
#             file_list[0].append(filepath)

#         # aishell-1
#         elif dataset == 'aishell-1':
#             if '.wav' not in file:
#                 continue
#             spk = int(root.split('/')[-1][1:])
#             filepath = os.path.join(root, file)
#             if spk >= 0 and spk < 100:
#                 file_list[0].append(filepath)
#             elif spk >= 100 and spk < 200:
#                 file_list[1].append(filepath)
#             elif spk >= 200 and spk < 300:
#                 file_list[2].append(filepath)
#             elif spk >= 300 and spk < 400:
#                 file_list[3].append(filepath)
#             elif spk >= 400 and spk < 500:
#                 file_list[4].append(filepath)
#             elif spk >= 500 and spk < 600:
#                 file_list[5].append(filepath)
#             elif spk >= 600 and spk < 700:
#                 file_list[6].append(filepath)
#             elif spk >= 700 and spk < 800:
#                 file_list[7].append(filepath)
#             elif spk >= 800 and spk < 900:
#                 file_list[7].append(filepath)
#             elif spk >= 900 and spk < 1000:
#                 file_list[7].append(filepath)
        
# for list in file_list:
#     print(len(list))
# process1 = multiprocessing.Process(target=extractor_pyworld, args=[file_list[0]])
# process2 = multiprocessing.Process(target=extractor_pyworld, args=[file_list[1]])
# process3 = multiprocessing.Process(target=extractor_pyworld, args=[file_list[2]])
# process4 = multiprocessing.Process(target=extractor_pyworld, args=[file_list[3]])
# process5 = multiprocessing.Process(target=extractor_pyworld, args=[file_list[4]])
# process6 = multiprocessing.Process(target=extractor_pyworld, args=[file_list[5]])
# process7 = multiprocessing.Process(target=extractor_pyworld, args=[file_list[6]])
# process8 = multiprocessing.Process(target=extractor_pyworld, args=[file_list[7]])

# process1.start()
# process2.start()
# process3.start()
# process4.start()
# process5.start()
# process6.start()
# process7.start()
# process8.start()

# process1.join()
# process2.join()
# process3.join()
# process4.join()
# process5.join()
# process6.join()
# process7.join()
# process8.join()