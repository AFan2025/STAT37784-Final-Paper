from .speechDataset import BrainToTextDataset, PhonemeDataset

def getDataset(datasetName):
    if datasetName == 'BrainToText':
        return BrainToTextDataset
    elif datasetName == 'Phoneme':
        return PhonemeDataset
    else:
        raise ValueError(f"Unknown dataset: {datasetName}")


__all__ = ["BrainToTextDataset", "PhonemeDataset", "getDataset"]