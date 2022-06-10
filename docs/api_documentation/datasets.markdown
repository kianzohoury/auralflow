---
layout: default
title: Datasets
parent: API Documentation
nav_order: 4
mathjax: true
---

# DATASETS

## CREATE AUDIO DATASET
Helper method that creates audio datasets.

<div class="doc-container-method" style="margin-left: 0px">
  <div class="doc-method" style="height: 60px">
    <div class="doc-label">METHOD</div>
    <div class="doc-label-multi">
      auralflow.datasets.create_audio_dataset(<i>dataset_path,
      targets, split='train', chunk_size=3, num_chunks=10000,
      max_num_tracks=None, sample_rate=44100, mono=True</i>)
    </div>
  </div>
</div>
Creates a chunked audio dataset.
<div class="doc-sub-container-method">
    <h4>Parameters</h4>
    <hr style="padding: 0px; margin: 0px; height: 2px">
    <ul>
      <li>
        <p>
          <i> dataset_path (str) </i> &nbsp; : &nbsp; Path to an audio dataset.
        </p>
      </li>
      <li>
        <p>
          <i> targets (List[str]) </i> &nbsp; : &nbsp; Target source labels.
        </p>
      </li>
       <li>
         <p>
           <i> split (str) </i> &nbsp; : &nbsp; Subset of dataset to read from. Default: 'train'.
         </p>
       </li>
       <li>
         <p>
           <i> chunk_size (int) </i> &nbsp; : &nbsp; Duration of each audio chunk. Default: 3.
         </p>
       </li>
       <li>
         <p>
           <i> num_chunks (int) </i> &nbsp; : &nbsp; Number of resampled audio chunks to create. Default: 1e4.
         </p>
       </li>
       <li>
         <p>
           <i> max_num_tracks (Optional[int]) </i> &nbsp; : &nbsp; Max number of tracks to resample from. Default: None.
         </p>
       </li>
       <li>
         <p>
           <i> sample_rate (int) </i> &nbsp; : &nbsp; Sample rate. Default: 44100.
         </p>
       </li>
       <li>
         <p>
           <i> mono (bool) </i> &nbsp; : &nbsp; Whether to sample tracks as mono or stereo. Default: True.
         </p>
       </li>
    </ul>
        <h4>Returns</h4>
            <hr style="padding: 0px; margin: 0px; height: 2px">
            <p>
              <i> (AudioDataset) </i> &nbsp; : &nbsp; Audio dataset.
            </p>
</div>



#### Example
```python
from auralflow.datasets import create_audio_dataset


# create 100,000 3-sec chunks from a pool of 80 total tracks
train_dataset = create_audio_dataset(
    dataset_path="path/to/dataset",
    split="train",
    targets=["vocals"],
    chunk_size=3,
    num_chunks=int(1e5),
    max_num_tracks=80,
    sample_rate=44100,
    mono=True,
)

# sample mixture and target training data from the dataset
mix_audio, target_audio = next(iter(train_dataset))
```


