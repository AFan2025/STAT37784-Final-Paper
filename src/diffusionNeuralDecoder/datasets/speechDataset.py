import numpy as np
import torch
from torch.utils.data import Dataset
import os

# PHONE_DEF = [
#     'AA', 'AE', 'AH', 'AO', 'AW',
#     'AY', 'B',  'CH', 'D', 'DH',
#     'EH', 'ER', 'EY', 'F', 'G',
#     'HH', 'IH', 'IY', 'JH', 'K',
#     'L', 'M', 'N', 'NG', 'OW',
#     'OY', 'P', 'R', 'S', 'SH',
#     'T', 'TH', 'UH', 'UW', 'V',
#     'W', 'Y', 'Z', 'ZH'
# ]

# PHONE_DEF_SIL = [
#     '<pad>','AA', 'AE', 'AH', 'AO', 'AW',
#     'AY', 'B',  'CH', 'D', 'DH',
#     'EH', 'ER', 'EY', 'F', 'G',
#     'HH', 'IH', 'IY', 'JH', 'K',
#     'L', 'M', 'N', 'NG', 'OW',
#     'OY', 'P', 'R', 'S', 'SH',
#     'T', 'TH', 'UH', 'UW', 'V',
#     'W', 'Y', 'Z', 'ZH', ' ', '<eos>'
# ]

PHONEMES = ['<pad>', '<unk>', '<s>', '</s>', ' ', 'AA0', 'AA1', 'AA2', 'AE0', 'AE1', 'AE2', 'AH0', 'AH1', 'AH2', 'AO0', 'AO1', 'AO2', 'AW0', 'AW1', 'AW2', 'AY0', 'AY1', 'AY2', 'B', 'CH', 'D', 'DH', 'EH0', 'EH1', 'EH2', 'ER0', 'ER1', 'ER2', 'EY0', 'EY1', 'EY2', 'F', 'G', 'HH', 'IH0', 'IH1', 'IH2', 'IY0', 'IY1', 'IY2', 'JH', 'K', 'L', 'M', 'N', 'NG', 'OW0', 'OW1', 'OW2', 'OY0', 'OY1', 'OY2', 'P', 'R', 'S', 'SH', 'T', 'TH', 'UH0', 'UH1', 'UH2', 'UW', 'UW0', 'UW1', 'UW2', 'V', 'W', 'Y', 'Z', 'ZH'] 
PHONE_TO_ID = {phone: idx for idx, phone in enumerate(PHONEMES)}
ID_TO_PHONE = {idx: phone for idx, phone in enumerate(PHONEMES)}

# PHONE_TO_ID = {phone: idx for idx, phone in enumerate(PHONE_DEF_SIL)}

# CHANG_PHONE_DEF = [
#     'AA', 'AE', 'AH', 'AW',
#     'AY', 'B',  'D', 'DH',
#     'EH', 'ER', 'EY', 'F', 'G',
#     'HH', 'IH', 'IY', 'K',
#     'L', 'M', 'N', 'NG', 'OW',
#     'P', 'R', 'S',
#     'T', 'TH', 'UH', 'UW', 'V',
#     'W', 'Y', 'Z'
# ]

# CONSONANT_DEF = ['CH', 'SH', 'JH', 'R', 'B',
#                  'M',  'W',  'V',  'F', 'P',
#                  'D',  'N',  'L',  'S', 'T',
#                  'Z',  'TH', 'G',  'Y', 'HH',
#                  'K', 'NG', 'ZH', 'DH']
# VOWEL_DEF = ['EY', 'AE', 'AY', 'EH', 'AA',
#              'AW', 'IY', 'IH', 'OY', 'OW',
#              'AO', 'UH', 'AH', 'UW', 'ER']

# SIL_DEF = ['SIL']

class BrainToTextDataset(Dataset):
    """
    PyTorch Dataset for Brain-to-Text competition data.
    Args:
        data_path (str): Path to the preprocessed data directory.
        partition (str): Data partition to use ('train', 'test', 'competitionHoldOut').

    Automatically detects whether the partition is stored as a single brain_data.npz
    or as shards described by a shard_manifest.txt.  For sharded partitions each shard
    is loaded into RAM on first access and cached, so no single allocation is ever the
    full partition size (which was the original OOM trigger at save time).
    """

    def __init__(self, data_path, partition='train'):
        self.data_path = data_path
        self.partition = partition
        part_dir = os.path.join(data_path, partition)
        self.vocab_size = len(PHONEMES)

        manifest_path = os.path.join(part_dir, 'shard_manifest.txt')
        if os.path.exists(manifest_path):
            self._sharded = True
            self._shard_paths = []
            shard_sizes = []
            with open(manifest_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    name, n = line.split('\t')
                    self._shard_paths.append(os.path.join(part_dir, name))
                    shard_sizes.append(int(n))
            # cumulative[i] = first global index belonging to shard i
            self._cumulative = np.concatenate([[0], np.cumsum(shard_sizes)]).astype(np.int64)
            self._total = int(self._cumulative[-1])
            # Cache: shard_idx -> dict of arrays. Populated lazily on first access.
            self._shard_cache: dict = {}
        else:
            self._sharded = False
            npz_path = os.path.join(part_dir, 'brain_data.npz')
            if not os.path.isfile(npz_path):
                raise FileNotFoundError(
                    f"No brain_data.npz or shard_manifest.txt found in {part_dir}"
                )
            data = np.load(npz_path, allow_pickle=True)
            self._input_features  = data['input_features']
            self._input_masks     = data['inputMasks']
            self._phoneme_tokens  = data['phoneme_tokens']
            self._phoneme_masks   = data['phoneme_masks']
            self._frame_lens      = data['frame_lens']
            self._transcriptions  = data['transcriptions']
            self._total           = len(self._input_features)

    def _get_shard(self, shard_idx: int) -> dict:
        """Load and cache a shard. Each shard is small enough to fit in RAM."""
        if shard_idx not in self._shard_cache:
            data = np.load(self._shard_paths[shard_idx], allow_pickle=True)
            self._shard_cache[shard_idx] = {
                'input_features': data['input_features'],
                'inputMasks':     data['inputMasks'],
                'phoneme_tokens': data['phoneme_tokens'],
                'phoneme_masks':  data['phoneme_masks'],
                'frame_lens':     data['frame_lens'],
                'transcriptions': data['transcriptions'],
            }
        return self._shard_cache[shard_idx]

    def __len__(self) -> int:
        return self._total

    def __getitem__(self, idx: int) -> dict:
        if self._sharded:
            # Binary search: find which shard owns this global index.
            shard_idx = int(np.searchsorted(self._cumulative[1:], idx, side='right'))
            local_idx = idx - int(self._cumulative[shard_idx])
            shard = self._get_shard(shard_idx)
            input_feat   = shard['input_features'][local_idx]
            input_mask   = shard['inputMasks'][local_idx]
            phoneme_tok  = shard['phoneme_tokens'][local_idx]
            phoneme_mask = shard['phoneme_masks'][local_idx]
            frame_len    = shard['frame_lens'][local_idx]
            transcription = str(shard['transcriptions'][local_idx])
        else:
            input_feat    = self._input_features[idx]
            input_mask    = self._input_masks[idx]
            phoneme_tok   = self._phoneme_tokens[idx]
            phoneme_mask  = self._phoneme_masks[idx]
            frame_len     = self._frame_lens[idx]
            transcription = str(self._transcriptions[idx])

        return {
            'input_features': torch.from_numpy(np.array(input_feat, dtype=np.float32)).permute(0,3,1,2),
            'input_mask':     torch.from_numpy(np.array(input_mask,  dtype=bool)),
            'phoneme_tokens': torch.from_numpy(np.array(phoneme_tok, dtype=np.int64)),
            'phoneme_mask':   torch.from_numpy(np.array(phoneme_mask, dtype=bool)),
            'frame_len':      torch.tensor(int(frame_len), dtype=torch.long),
            'transcription':  transcription,
        }

class PhonemeDataset(Dataset):
    """Dataset for pre-computed phoneme sequences"""
    
    def __init__(self, data_path):
        if not os.path.isfile(data_path):
            raise FileNotFoundError(f"Phoneme dataset not found at {data_path}")

        data = np.load(data_path, allow_pickle=True)

        required_keys = {'phoneme_data', 'phoneme_mask', 'phoneme_to_id', 'max_phoneme_len'}
        missing_keys = required_keys.difference(data.files)
        if missing_keys:
            raise KeyError(f"Missing keys in phoneme dataset: {sorted(missing_keys)}")

        self.data_path = data_path
        self.phoneme_data = data['phoneme_data']
        self.phoneme_mask = data['phoneme_mask']
        self.phoneme_to_id = data['phoneme_to_id'].item()
        self.max_phoneme_len = int(data['max_phoneme_len'].item())
        self.pad_token_id = self.phoneme_to_id['<pad>']
        self.vocab_size = len(self.phoneme_to_id)

        if len(self.phoneme_data) != len(self.phoneme_mask):
            raise ValueError("phoneme_data and phoneme_mask must have the same number of rows")
    
    def __len__(self):
        return len(self.phoneme_data)
    
    def __getitem__(self, idx):
        input_ids = torch.from_numpy(self.phoneme_data[idx]).long()
        attention_mask = torch.from_numpy(self.phoneme_mask[idx]).bool()
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            # 'phonemes': input_ids,
            # 'mask': attention_mask,
        }

# Usage
# dataset = PhonemeDataset('../../../preprocessed_data/phoneme_data.npz')