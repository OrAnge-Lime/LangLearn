import time
import torch
from glob import glob

device = torch.device('gpu')  # gpu also works, but our models are fast enough for CPU

silero_model, silero_decoder, utils = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                       model='silero_stt',
                                       language='en',
                                       device=device)
(read_batch, split_into_batches,
 read_audio, prepare_model_input) = utils  # see function signature for details

def silero_predict(path):
    test_files = glob(path)
    batches = split_into_batches(test_files, batch_size=10)
    input = prepare_model_input(read_batch(batches[0]),
                                device=device)

    output = silero_model(input)
    result = ""
    for example in output:
        result += silero_decoder(example.cpu())
    return result