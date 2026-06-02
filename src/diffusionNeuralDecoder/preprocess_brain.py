import scipy.io
import numpy as np
from g2p_en import G2p
import os
from diffusionNeuralDecoder.datasets.speechDataset import PHONE_TO_ID
from dotenv import load_dotenv
import logging
import re
import nltk

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

load_dotenv()

COMPETITION_DATA_DIR = os.getenv('COMPETITION_DATA_DIR', '/data/CSMC_35200_BraintoTextGroup/competitionData')
PREPROCESSED_DATA_DIR = os.getenv('PREPROCESSED_DATA_DIR', '/data/CSMC_35200_BraintoTextGroup/preprocessed_data')

ROWS = np.array([
    [62, 60, 63, 58, 59, 61, 56, 57, 125, 123, 121, 119, 117, 115, 113, 127],
    [51, 53, 54, 55, 45, 49, 52, 50, 126, 124, 122, 120, 118, 116, 114, 111],
    [43, 41, 47, 48, 46, 42, 39, 37, 112, 110, 109, 108, 107, 106, 105, 104],
    [35, 33, 44, 40, 38, 36, 34, 32, 103, 102, 101, 100, 99, 97, 98, 96],
    [94, 95, 93, 92, 91, 90, 89, 88, 31, 29, 27, 25, 23, 21, 17, 30],
    [87, 86, 84, 85, 82, 83, 81, 80, 28, 26, 19, 15, 13, 20, 24, 22],
    [79, 77, 75, 73, 71, 69, 67, 65, 11, 9, 18, 12, 10, 7, 14, 16, ],
    [78, 76, 74, 72, 70, 68, 66, 64, 8, 5, 4, 6, 3, 2, 0, 1]
]).T #originally columns but rows seem easier for indexing, Shape (16, 8)

TOLERABLE_SEQ_LEN = os.getenv('TOLERABLE_SEQ_LEN', None)  #extra tolerance length when finding max sequence length for brain data
TOLERABLE_SEQ_PERCENTILE = float(os.getenv('TOLERABLE_SEQ_PERCENTILE', 95))  #percentile for tolerable sequence length if MAX_SEQ_LEN not set
MAX_PHONEME_LEN = int(os.getenv("MAX_PHONEME_LEN", 128))

logger.info(f"tolerable sequence len provided is {TOLERABLE_SEQ_LEN}")
logger.info(f"tolerable sequence percentage is {TOLERABLE_SEQ_PERCENTILE}")
logger.info(f"max phoneme len is {MAX_PHONEME_LEN}")

def ensure_nltk_data():
    """Download NLTK data if not already present."""
    try:
        nltk.data.find('taggers/averaged_perceptron_tagger_eng')
    except LookupError:
        print("Downloading required NLTK data...")
        nltk.download('averaged_perceptron_tagger_eng', quiet=True)
        nltk.download('punkt', quiet=True)

#G2p requires an additional nltk data download
ensure_nltk_data()
G2P_ENGINE = G2p()

def find_max_seq_len(competition_data_dir=COMPETITION_DATA_DIR, tolerable_len=None, tolerable_percentile: float =95):
    """
    Find the maximum sequence length across all .mat files in the Brain-to-Text competition dataset.
    Assumes that data is organized in train, test, and competitionHoldOut folders.
    Args:
        competition_data_dir (str): Path to the competition data directory.
        tolerable_len (int, optional): Extra tolerance length when finding max sequence length.
        tolerable_percentile (int, optional): Percentile for tolerable sequence length if tolerable_len is not set.
    Returns:
        int: Maximum sequence length found.
        list: List of all sequence lengths found.
    """
    logger.info("Finding maximum sequence length in dataset...")
    seq_lengths = []

    for division in ["train", "test", "competitionHoldOut"]:
        names_list = os.listdir(os.path.join(competition_data_dir, division))
        for name in names_list:
            if name[-3:] == "txt":
                logger.info(f"skipping {name}")
                continue
            data_path = os.path.join(competition_data_dir, division, name)
            dat = scipy.io.loadmat(data_path)
            # features = np.concatenate([dat['tx1'][0,i][:,0:128], dat['spikePow'][0,i][:,0:128]], axis=1)
            for i in range(dat['sentenceText'].shape[0]):    
                input_features = dat['tx1'][0,i] # Shape: (num_time_steps, 256)
                seq_length = input_features.shape[0]
                seq_lengths.append(seq_length)
    if tolerable_len is not None:
        max_seq_len = tolerable_len
    else:
        max_seq_len = int(np.percentile(seq_lengths, tolerable_percentile))
    return max_seq_len, seq_lengths

#TO DO for outputting files
def preprocess_1D(sessionName, dataPath, max_seq_len, outputFolder):
    """
    Preprocess raw .mat data files into .pt files format for Brain-to-Text competition.
    This is the step for the model architecture that diregards spatial features and only uses 1D features like tx1 or spike power along the 256 flattented feature space.
    The output for each datapoint will be (T, 256)
    """
    partNames = ['train','test','competitionHoldOut']
    
    for partIdx in range(len(partNames)):
        sessionPath = dataPath + '/' + partNames[partIdx] + '/' + sessionName + '.mat'
        if not os.path.isfile(sessionPath):
            continue
            
        dat = scipy.io.loadmat(sessionPath)

        input_features = []
        transcriptions = []
        frame_lens = []
        n_trials = dat['sentenceText'].shape[0]

        #collect area 6v tx1 and spikePow features
        for i in range(n_trials):    
            #get time series of TX and spike power for this trial
            #first 128 columns = area 6v only
            features = np.concatenate([dat['tx1'][0,i][:,0:128], dat['spikePow'][0,i][:,0:128]], axis=1)

            sentence_len = features.shape[0]
            sentence = dat['sentenceText'][i].strip()

            input_features.append(features)
            transcriptions.append(sentence)
            frame_lens.append(sentence_len)

        #block-wise feature normalization
        blockNums = np.squeeze(dat['blockIdx'])
        blockList = np.unique(blockNums)
        blocks = []
        for b in range(len(blockList)):
            sentIdx = np.argwhere(blockNums==blockList[b])
            sentIdx = sentIdx[:,0].astype(np.int32)
            blocks.append(sentIdx)

        for b in range(len(blocks)):
            feats = np.concatenate(input_features[blocks[b][0]:(blocks[b][-1]+1)], axis=0)
            feats_mean = np.mean(feats, axis=0, keepdims=True)
            feats_std = np.std(feats, axis=0, keepdims=True)
            for i in blocks[b]:
                input_features[i] = (input_features[i] - feats_mean) / (feats_std + 1e-8)

        #convert to tfRecord file
        session_data = {
            'inputFeatures': input_features,
            'transcriptions': transcriptions,
            'frameLens': frame_lens
        }

def preprocess_2D(dataPath, outputFolder, max_seq_len = 512, max_phoneme_len=128):
    """
    Preprocess raw .mat data files into .pt files format for Brain-to-Text competition.
    This is the step for the model architecture that uses the convolutional layers to process spatial features
    The output for each datapoint will be (T,H,W,2) where 2 channels are tx1 and spike power respectively.
    This is different from the 1D feature preprocessing where the output is (T, feature_dim)
    """
    
    partNames = ['test','competitionHoldOut']
    
    for partIdx in partNames:

        output_dir = os.path.join(outputFolder, partIdx)
        os.makedirs(output_dir, exist_ok=True)

        # Flat lists are much faster to stack than nested session-level object lists.
        total_input_features = []
        total_inputMasks = []
        total_phoneme_tokens = []
        total_phoneme_masks = []
        total_transcriptions = []
        total_frame_lens = []

        part_dir = os.path.join(dataPath, partIdx)
        names_list = sorted(os.listdir(part_dir))
        for sessionName in names_list:
            if sessionName.endswith(".txt"):
                logger.info(f"skipping {sessionName}")
                continue

            sessionPath = os.path.join(part_dir, sessionName)
            if not os.path.isfile(sessionPath):
                continue

            dat = scipy.io.loadmat(sessionPath)
            blockNums = np.squeeze(dat['blockIdx']).astype(np.int32)
            n_trials = dat['sentenceText'].shape[0]

            session_records = []

            # Collect valid trials first so normalization only touches kept examples.
            for i in range(n_trials):
                tx1 = dat['tx1'][0, i][:, 0:128]
                spikePow = dat['spikePow'][0, i][:, 0:128]

                assert tx1.shape[0] == spikePow.shape[0]
                seq_len = tx1.shape[0]
                if seq_len >= max_seq_len:
                    continue

                sentence = str(np.squeeze(dat['sentenceText'][i])).strip()
                phoneme_ids = g2p_transcription(sentence)

                phoneme_array = np.full((max_phoneme_len,), PHONE_TO_ID['<pad>'], dtype=np.int16)
                phoneme_mask = np.zeros((max_phoneme_len,), dtype=np.bool_)
                valid_phoneme_len = min(len(phoneme_ids), max_phoneme_len)
                if valid_phoneme_len > 0:
                    phoneme_array[:valid_phoneme_len] = np.asarray(phoneme_ids[:valid_phoneme_len], dtype=np.int16)
                    phoneme_mask[:valid_phoneme_len] = True

                session_records.append({
                    'tx1': tx1,
                    'spikePow': spikePow,
                    'sentence': sentence,
                    'phoneme_tokens': phoneme_array,
                    'phoneme_mask': phoneme_mask,
                    'frame_len': seq_len,
                    'block': int(blockNums[i]),
                })

            if not session_records:
                logger.info(f"No valid trials kept for session {sessionName} in partition {partIdx}")
                continue

            logger.info(f'Normalizing features block-wise for session {sessionName} in partition {partIdx}')
            block_to_record_idxs = {}
            for rec_idx, rec in enumerate(session_records):
                block_to_record_idxs.setdefault(rec['block'], []).append(rec_idx)

            for record_idxs in block_to_record_idxs.values():
                block_tx_features = np.concatenate([session_records[r]['tx1'] for r in record_idxs], axis=0)
                tx_mean = np.mean(block_tx_features, axis=0, keepdims=True)
                tx_std = np.std(block_tx_features, axis=0, keepdims=True)

                block_spike_features = np.concatenate([session_records[r]['spikePow'] for r in record_idxs], axis=0)
                spike_mean = np.mean(block_spike_features, axis=0, keepdims=True)
                spike_std = np.std(block_spike_features, axis=0, keepdims=True)

                for r in record_idxs:
                    session_records[r]['tx1'] = (session_records[r]['tx1'] - tx_mean) / (tx_std + 1e-8)
                    session_records[r]['spikePow'] = (session_records[r]['spikePow'] - spike_mean) / (spike_std + 1e-8)

            logger.info(f'Reshaping and padding features for session {sessionName} in partition {partIdx}')
            for rec in session_records:
                tx1_map = rec['tx1'][:, ROWS]
                spike_map = rec['spikePow'][:, ROWS]
                feature = np.stack([tx1_map, spike_map], axis=-1).astype(np.float16, copy=False)

                seq_len = rec['frame_len']
                padded_feature = np.zeros((max_seq_len, 16, 8, 2), dtype=np.float16)
                padded_feature[:seq_len] = feature

                input_mask = np.zeros((max_seq_len,), dtype=np.bool_)
                input_mask[:seq_len] = True

                total_input_features.append(padded_feature)
                total_inputMasks.append(input_mask)
                total_phoneme_tokens.append(rec['phoneme_tokens'])
                total_phoneme_masks.append(rec['phoneme_mask'])
                total_transcriptions.append(rec['sentence'])
                total_frame_lens.append(seq_len)

            logger.info(f'Preprocessed {len(session_records)} trials for session {sessionName} in partition {partIdx}')

        if total_input_features:
            total_input_features = np.stack(total_input_features, axis=0)
            total_inputMasks = np.stack(total_inputMasks, axis=0)
            total_phoneme_tokens = np.stack(total_phoneme_tokens, axis=0)
            total_phoneme_masks = np.stack(total_phoneme_masks, axis=0)
            total_transcriptions = np.array(total_transcriptions, dtype=object)
            total_frame_lens = np.asarray(total_frame_lens, dtype=np.int32)
        else:
            total_input_features = np.zeros((0, max_seq_len, 16, 8, 2), dtype=np.float16)
            total_inputMasks = np.zeros((0, max_seq_len), dtype=np.bool_)
            total_phoneme_tokens = np.zeros((0, max_phoneme_len), dtype=np.int32)
            total_phoneme_masks = np.zeros((0, max_phoneme_len), dtype=np.bool_)
            total_transcriptions = np.array([], dtype=object)
            total_frame_lens = np.zeros((0,), dtype=np.int32)

        logger.info(f"Processed {len(total_input_features)} brain sequences.")

        output_path = os.path.join(output_dir, "brain_data.npz")
        np.savez(
            output_path,
            input_features=total_input_features,
            inputMasks=total_inputMasks,
            phoneme_tokens=total_phoneme_tokens,
            phoneme_masks=total_phoneme_masks,
            transcriptions=total_transcriptions,
            frame_lens=total_frame_lens,
            max_phoneme_len=max_phoneme_len,
        )
        logger.info(f"Brain preprocessing for {partIdx} completed. Data saved to {output_path}")

def g2p_transcription(sentence):
    """
    Convert a sentence into its phoneme representation using g2p_en.
    Args:
        sentence (str): Input sentence.
    Returns:
        list: List of phonemes.
    """
    tokenized_sentence = []
    sentence = re.sub(r'[^a-zA-Z\- \']', '', sentence)  # Remove punctuation except hyphens and apostrophes
    sentence = sentence.replace('--', '').lower()
    phonemes = G2P_ENGINE(sentence)
    phonemes.append('</s>')
    phonemes.insert(0,'<s>')
    for phoneme in phonemes:
        if phoneme not in PHONE_TO_ID:
            logger.warning(f'Phoneme {phoneme} not in PHONE_TO_ID mapping.')
        else:
            tokenized_sentence.append(PHONE_TO_ID[phoneme])
    return tokenized_sentence

def preprocess_2D_sharded(partitions: list,
                          dataPath, 
                          outputFolder, 
                          max_seq_len=512, 
                          max_phoneme_len=128, 
                          shard_size=500):
    """
    Preprocess and shard large partitions (e.g. train) into multiple .npz files.
    Each shard contains at most shard_size trials. Shards are flushed only after a
    complete session is processed so that no block is ever split across shards
    (block-wise normalization is fully applied within a session before any write).

    Output files are named:  <outputFolder>/<partIdx>/brain_data_shard_NNNN.npz
    A manifest file          <outputFolder>/<partIdx>/shard_manifest.txt
    is also written listing each shard path and its trial count, one line per shard:
        brain_data_shard_0000.npz\t<n_trials>
    """

    for partIdx in partitions:
        output_dir = os.path.join(outputFolder, partIdx)
        os.makedirs(output_dir, exist_ok=True)

        # Per-shard accumulation buffers.
        buf_input_features = []
        buf_inputMasks = []
        buf_phoneme_tokens = []
        buf_phoneme_masks = []
        buf_transcriptions = []
        buf_frame_lens = []
        shard_idx = 0
        manifest_entries = []

        def _flush_shard():
            nonlocal shard_idx
            if not buf_input_features:
                return
            arr_input      = np.stack(buf_input_features, axis=0)
            arr_masks      = np.stack(buf_inputMasks, axis=0)
            arr_phoneme    = np.stack(buf_phoneme_tokens, axis=0)
            arr_ph_masks   = np.stack(buf_phoneme_masks, axis=0)
            arr_text       = np.array(buf_transcriptions, dtype=object)
            arr_frame_lens = np.asarray(buf_frame_lens, dtype=np.int32)

            shard_name = f"brain_data_shard_{shard_idx:04d}.npz"
            out_path   = os.path.join(output_dir, shard_name)
            np.savez(
                out_path,
                input_features=arr_input,
                inputMasks=arr_masks,
                phoneme_tokens=arr_phoneme,
                phoneme_masks=arr_ph_masks,
                transcriptions=arr_text,
                frame_lens=arr_frame_lens,
                max_phoneme_len=max_phoneme_len,
            )
            manifest_entries.append((shard_name, len(buf_input_features)))
            logger.info(
                f"[{partIdx}] Saved shard {shard_idx:04d} "
                f"({len(buf_input_features)} trials) → {out_path}"
            )
            buf_input_features.clear()
            buf_inputMasks.clear()
            buf_phoneme_tokens.clear()
            buf_phoneme_masks.clear()
            buf_transcriptions.clear()
            buf_frame_lens.clear()
            shard_idx += 1

        part_dir   = os.path.join(dataPath, partIdx)
        names_list = sorted(os.listdir(part_dir))

        for sessionName in names_list:
            if sessionName.endswith(".txt"):
                logger.info(f"skipping {sessionName}")
                continue

            sessionPath = os.path.join(part_dir, sessionName)
            if not os.path.isfile(sessionPath):
                continue

            dat       = scipy.io.loadmat(sessionPath)
            blockNums = np.squeeze(dat['blockIdx']).astype(np.int32)
            n_trials  = dat['sentenceText'].shape[0]

            session_records = []

            # Collect valid trials for this session.
            for i in range(n_trials):
                tx1      = dat['tx1'][0, i][:, 0:128]
                spikePow = dat['spikePow'][0, i][:, 0:128]

                assert tx1.shape[0] == spikePow.shape[0]
                seq_len = tx1.shape[0]
                if seq_len >= max_seq_len:
                    continue

                sentence    = str(np.squeeze(dat['sentenceText'][i])).strip()
                phoneme_ids = g2p_transcription(sentence)

                phoneme_array = np.full((max_phoneme_len,), PHONE_TO_ID['<pad>'], dtype=np.int16)
                phoneme_mask  = np.zeros((max_phoneme_len,), dtype=np.bool_)
                valid_phoneme_len = min(len(phoneme_ids), max_phoneme_len)
                if valid_phoneme_len > 0:
                    phoneme_array[:valid_phoneme_len] = np.asarray(phoneme_ids[:valid_phoneme_len], dtype=np.int16)
                    phoneme_mask[:valid_phoneme_len]  = True

                session_records.append({
                    'tx1':            tx1,
                    'spikePow':       spikePow,
                    'sentence':       sentence,
                    'phoneme_tokens': phoneme_array,
                    'phoneme_mask':   phoneme_mask,
                    'frame_len':      seq_len,
                    'block':          int(blockNums[i]),
                })

            if not session_records:
                logger.info(f"No valid trials kept for session {sessionName} in partition {partIdx}")
                continue

            # Block-wise normalization — must complete before any write.
            logger.info(f'Normalizing features block-wise for session {sessionName} in partition {partIdx}')
            block_to_record_idxs: dict = {}
            for rec_idx, rec in enumerate(session_records):
                block_to_record_idxs.setdefault(rec['block'], []).append(rec_idx)

            for record_idxs in block_to_record_idxs.values():
                block_tx    = np.concatenate([session_records[r]['tx1']      for r in record_idxs], axis=0)
                tx_mean     = np.mean(block_tx, axis=0, keepdims=True)
                tx_std      = np.std(block_tx,  axis=0, keepdims=True)

                block_spike = np.concatenate([session_records[r]['spikePow'] for r in record_idxs], axis=0)
                spike_mean  = np.mean(block_spike, axis=0, keepdims=True)
                spike_std   = np.std(block_spike,  axis=0, keepdims=True)

                for r in record_idxs:
                    session_records[r]['tx1']      = (session_records[r]['tx1']      - tx_mean)    / (tx_std    + 1e-8)
                    session_records[r]['spikePow']  = (session_records[r]['spikePow'] - spike_mean) / (spike_std + 1e-8)

            # Reshape, pad, and append to the current shard buffer.
            logger.info(f'Reshaping and padding features for session {sessionName} in partition {partIdx}')
            for rec in session_records:
                tx1_map   = rec['tx1'][:, ROWS]
                spike_map = rec['spikePow'][:, ROWS]
                feature   = np.stack([tx1_map, spike_map], axis=-1).astype(np.float16, copy=False)

                seq_len        = rec['frame_len']
                padded_feature = np.zeros((max_seq_len, 16, 8, 2), dtype=np.float16)
                padded_feature[:seq_len] = feature

                input_mask           = np.zeros((max_seq_len,), dtype=np.bool_)
                input_mask[:seq_len] = True

                buf_input_features.append(padded_feature)
                buf_inputMasks.append(input_mask)
                buf_phoneme_tokens.append(rec['phoneme_tokens'])
                buf_phoneme_masks.append(rec['phoneme_mask'])
                buf_transcriptions.append(rec['sentence'])
                buf_frame_lens.append(seq_len)

            logger.info(
                f'Preprocessed {len(session_records)} trials for session {sessionName} in partition {partIdx} '
                f'(shard buffer: {len(buf_input_features)} / {shard_size})'
            )

            # Flush once the buffer reaches the shard size. We only do this at a
            # session boundary so no block is ever divided between two shards.
            if len(buf_input_features) >= shard_size:
                _flush_shard()

        # Flush any remaining trials in the buffer.
        _flush_shard()

        # Write the manifest so downstream code can discover shards without glob.
        manifest_path = os.path.join(output_dir, "shard_manifest.txt")
        with open(manifest_path, "w") as f:
            for shard_name, n in manifest_entries:
                f.write(f"{shard_name}\t{n}\n")
        total_trials = sum(n for _, n in manifest_entries)
        logger.info(
            f"Brain preprocessing for {partIdx} completed. "
            f"{shard_idx} shards, {total_trials} total trials. "
            f"Manifest: {manifest_path}"
        )


if __name__ == "__main__":
    if TOLERABLE_SEQ_LEN is None:
        logger.info(f"Default tolerable sequence length not found, using percentile: {TOLERABLE_SEQ_PERCENTILE}")
        max_sequence_len, seq_lengths = find_max_seq_len(COMPETITION_DATA_DIR,
                                                        tolerable_len=TOLERABLE_SEQ_LEN,
                                                        tolerable_percentile=TOLERABLE_SEQ_PERCENTILE)
        logger.info(f'Determined max_seq_len: {max_sequence_len}')
    else:
        max_sequence_len = int(TOLERABLE_SEQ_LEN)
        logger.info(f'Using provided tolerable_seq_len: {max_sequence_len}')

    #DON'T NEED TO RUN AGAIN, COMPETITION HOLD OUT AND TEST SET ARE GOOD, NEED TO CREATE A WHOLE NEW METHOD FOR THE TRAINING SHARDS DUE TO SIZE
    # preprocess_2D(
    #     dataPath=COMPETITION_DATA_DIR,
    #     outputFolder=PREPROCESSED_DATA_DIR,
    #     max_seq_len=max_sequence_len,
    #     max_phoneme_len=MAX_PHONEME_LEN,
    # )
    preprocess_2D_sharded(["train"],
                          dataPath=COMPETITION_DATA_DIR,
                          outputFolder=PREPROCESSED_DATA_DIR, 
                          max_seq_len=max_sequence_len, 
                          max_phoneme_len=MAX_PHONEME_LEN, 
                          shard_size=500)
    