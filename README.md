# Background-Domain-Switch (BDS) for Robust SED

An implementation example for adopting the BDS data augmentation approach proposed in this paper (paper link: https://www.isca-archive.org/interspeech_2023/lin23_interspeech.pdf). The work is presented and published in Interspeech'23. This repository will demonstrate on how to apply BDS using [DCASE Challenge-Task4](https://dcase.community/challenge2022/index) as example.


## Purpose of this Software

This software is a research prototype, solely developed for and published as part of the publication mentioned above. It will neither be maintained nor monitored in any way.


## Running Environment

Please refer to the [official baseline recipe](https://github.com/DCASE-REPO/DESED_task/tree/master/recipes/dcase2022_task4_baseline) provided by DCASE website.


## How to Use

Simply adding the BDS function to pytorch lightning trainer step prior to other data augmentation approaches!
e.g., inside **/local/sed_trainer.py** 

```python
...
from desed_task.data_augm import BDS

...
	def training_step(self, batch, batch_indx):
		...

		# deriving masks for each dataset
		strong_mask = torch.zeros(batch_num).to(features).bool()
		weak_mask = torch.zeros(batch_num).to(features).bool()
		unlabeled_mask = torch.zeros(batch_num).to(features).bool()
		strong_mask[:indx_synth] = 1
		weak_mask[indx_synth : indx_weak + indx_synth] = 1
		unlabeled_mask[indx_weak + indx_synth:] = 1
		
		# BDS has to apply after some training epochs for reliable pseudo-labeling results
		if self.current_epoch/self.hparams["training"]["n_epochs"]>=0.6: # e.g., apply at latest 40% epochs
			features = BDS(feats=features,
						norm=self.scaler,
						scale=self.take_log,
						labels_strong=labels,
						set_masks=[strong_mask, weak_mask, unlabeled_mask],
						model=self.sed_student,
						seq_pooling_factor=4,
						event_threshold=0.4,
						min_frames=40,
						bidirectional=False,
						stochastic_iter=1)
		
		# other data augmentations, e.g., MixUp, SpecAugment...etc
		...
```


## Cite

If you use this code, please cite the following paper:

Wei-Cheng Lin, Luca Bondi and Shabnam Ghaffarzadegan, "Background Domain Switch: A Novel Data Augmentation Technique for Robust Sound Event Detection", Interspeech 2023.
```
@InProceedings{LinBDS_2023, 
  author={W.-C. Lin and L. Bondi and S. Ghaffarzadegan}, 
  title={{Background Domain Switch}: A Novel Data Augmentation Technique for Robust Sound Event Detection},
  booktitle={Interspeech 2023}, 
  volume={},
  year={2023}, 
  month={August}, 
  pages={326--330}, 
  address =  {Dublin, Ireland},
  doi={10.21437/Interspeech.2023-176},
}
```


## License

The code in this repository is open-sourced under the AGPL-3.0 license. See the [LICENSE](./LICENSE) file for details.

