---
layout: default
title: Separating Audio Files
parent: Basic Usage
nav_order: 3
---

# Separating Audio <a name="separating-audio"></a>
The separation script allows us to separate one or more songs at a time and
save the results locally. To separate audio we simply use the `separate`
command, which will build and load the best performing version of a model,
and run inference with it.

After, the results are placed within a single folder called `separated_audio`
to a desired location like so: 

```bash
path/to/separated_audio
  └── artist - track_name
        ├── original.wav
        ├── vocals.wav
        └── residual.wav
```
where each track gets its own folder containing the original track as well as
up to 4 separated stems *or* up to 3 separated stems plus the residual track.

# separate
## Separating a Single Music Track
To separate a single music track, we must specify the path to a song file
as well as the destination we wish to save the separated audio files to 
(defaults to current directory). 
```bash
auralflow separate my_model path/to/song --save to/this/path
```

## Separating a Folder of Music Tracks
Likewise, we can separate a folder of one or more music tracks by instead
passing in a valid directory of audio files as `path/to/folder`.
```bash
auralflow separate my_model path/to/folder --save to/this/path
```

## Additional Commands
We can also specify the following:
* ## `--duration`
***int*** : The max duration of each separated audio track.
* ## `--residual`
    Whether to save the residual track (only applies to models that separate <4 sources).
  
[Previous: Training Models](training.html){: .btn .btn-outline }
  {: .float-left}
