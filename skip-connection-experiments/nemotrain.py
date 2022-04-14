import argparse
import nemo.collections.asr as nemo_asr
from nemo.utils.exp_manager import exp_manager
import pytorch_lightning as pl
from omegaconf import DictConfig


def load(config):
  try:
      from ruamel.yaml import YAML
  except ModuleNotFoundError:
      from ruamel_yaml import YAML
  yaml = YAML(typ='safe')
  with open(config) as f:
      params = yaml.load(f)
  return params


def inference( params, first_asr_model ):
    first_asr_model.setup_test_data(test_data_config=params['model']['validation_ds'])
    first_asr_model.cuda()
     # We will be computing Word Error Rate (WER) metric between our hypothesis and predictions.
     # WER is computed as numerator/denominator.
     # We'll gather all the test batches' numerators and denominators.
    wer_nums = []
    wer_denoms = []

     # Loop over all test batches.
     # Iterating over the model's `test_dataloader` will give us:
     # (audio_signal, audio_signal_length, transcript_tokens, transcript_length)
     # See the AudioToCharDataset for more details.
    for test_batch in first_asr_model.test_dataloader():
        test_batch = [x.cuda() for x in test_batch]
        targets = test_batch[2]
        targets_lengths = test_batch[3]        
        log_probs, encoded_len, greedy_predictions = first_asr_model(input_signal=test_batch[0], input_signal_length=test_batch[1])
        # Notice the model has a helper object to compute WER
        first_asr_model._wer.update(greedy_predictions, targets, targets_lengths)
        _, wer_num, wer_denom = first_asr_model._wer.compute()
        first_asr_model._wer.reset()
        wer_nums.append(wer_num.detach().cpu().numpy())
        wer_denoms.append(wer_denom.detach().cpu().numpy())
     # Release tensors from GPU memory
        del test_batch, log_probs, targets, targets_lengths, encoded_len, greedy_predictions
          # We need to sum all numerators and denominators first. Then divide.
    print(f"WER = {sum(wer_nums)/sum(wer_denoms)}")


def main():
  parser = argparse.ArgumentParser(description="QuartzNet skip connection testing for efficient FPGA implementation")
  parser.add_argument('--model-config', '-c', required=True, help='yaml config of desired model')
  args = parser.parse_args()

  # data_dir ='/app/dev-nemo/data'
  # train_manifest = data_dir + '/train_clean_5.json'
  # test_manifest = data_dir + '/dev_clean_2.json'
  # #load the data:
  params = load(args.model_config)
  trainer = pl.Trainer(gpus=1)

  #use the model:
  # params['model']['train_ds']['manifest_filepath'] = train_manifest
  # params['model']['validation_ds']['manifest_filepath'] = test_manifest
  first_asr_model = nemo_asr.models.EncDecCTCModel(cfg=DictConfig(params['model']), trainer=trainer)

  #training starts:
  trainer.fit(first_asr_model)
  inference(params , first_asr_model)
    



if __name__ =='__main__':
  main()

