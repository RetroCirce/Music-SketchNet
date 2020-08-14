# Music-SketchNet Paper Repository 
Music SketchNet: Controllable Music Generation via Factorized Representations of Pitch and Rhythm, ISMIR 2020 <br>
* [Paper Link](https://arxiv.org/abs/2008.01291) <br>
* Some samples in this [link](https://drive.google.com/drive/folders/1CI__Tts_YUyHCjnunqyIHVrA-IasCgUq?usp=sharing).<br>
* We plan to provide a website for you to create you own sketch music (--in construction--). <br>
* Feel free to ask any question to [knutchen@ucsd.edu](mailto:knutchen@ucsd.edu), if you meet trouble in these codes.
## Intro
Music SketchNet allows you to specify your own music ideas, namely pitch contour and rhythm, in the monophonic music generation scenario.
It contains:
* SketchVAE converts the music meldoy tokens into the disentanglemd latent vector (z_pitch, z_rhythm)
* SketchInpainter receives the music latent vectors to predict the missing measure vector separately in rhythm and pitch space
* SketchConnector incorpoates a transformer encoder structure and random unmasking scheme in receiving the sketch control and finalizing the generation.

<p align="center">
<img src="figs/figure_1.png" align="center" alt="The Music Sketch Scenario" width="60%"/>
</p>

## Score Demo
Music SketchNet allows you to control the pitch, control the rhythm, or control both information in a music inpainting/completion scenario.
![The Score Demo](figs/figure_7.png)

## Code Guidance
We provide a complete jupyter notebook tutorial/pipline to guide you into each step of the Music SketchNet:
* Data Processing
* VAE Models Constrution, Training, and Inference
* SketchNet Construction, Training, Inference, and Comarison with Other Models
* Experiments, including Model Metric Evaluation, VAE Independent Evaluation, Significance Test (in preparation), and Sketch Quantitaive Analysis

To run the code, you need to down the extra material data from this [link](https://drive.google.com/drive/folders/1xmg4ijhVJTse-V2BQ1hEBG2hQGYDQjTp?usp=sharing)
> data.zip: all processed data npy files, if you want to go through all process, you need to download it.

> response.csv: the subjective listening test, if you want to go through all process, you need to download it.

> generation_result_npy.zip: the generation results from our training in 4-3, 4-4, 4-5, we highly recommend you to download it to let you get familiar with the naming and process of the experiement.

> IrishFolkSong.zip: if you only want to train and test three VAE structures (from 1-1 to 2-3), you only need to download this basic music midi data

> model_backup.zip: if you only want to skip most process and test the model, you can download the saved parameters for all models we trained.

#### Quick Start: 
> click the file [1-irish-process.ipynb](/1-irish-process.ipynb) follow from *1-irish-process.ipynb* to *5-3-bootstrap-sig-test.ipynb*

In each folder, we provide the model python script, or the data loader. And it should be easily recognized by the folder name.
Below is the introduction of all jupyter notebook files:
* 1-irish-process.ipynb: process the irish folk song midi data for the later use in VAE trainings.
* 2-1-EC2VAE.ipynb: train EC2-VAE.
* 2-2-MeasureVAE.ipynb: train MeasureVAE.
* 2-3-1-SketchVAE-data-process.ipynb: Split the data into the format for the  SketchVAE training.
* 2-3-2-SketchVAE-train.ipynb: train SketchVAE.
* 3-1-MusicInpaintNet.ipynb: train Music InpaintNet
* 3-2-SketchVAE_InpaintRNN.ipynb: train SketchVAE + InpaintRNN
* 3-3-SketchVAE_SketchInpainter.ipynb: train SketchInapinter (stage I training)
* 3-4-SketchNet.ipynb: train SketchNet (stage II training)
* 4-1 is deprecated
* 4-2-reptition-nonrepetition-dataset.ipynb: create sub-test set Irish-Test-R / NR
* 4-3-MusicInpaintNet-eval.ipynb: infer/output the generation from Music InpaintNet
* 4-4-SketchVAE-InpaintRNN-eval.ipynb: infer/output the generation from SketchVAE+InpaintRNN
* 4-5-SketchNet-eval.ipynb: infer/output the generation from SketchNet
* 5-1-loss_and_accuracy.ipynb: experiement concerned with the loss and the accuracy.
* 5-2-vae_independent_test.ipynb: Three VAE structures independent test
* 5-3-bootstrap-sig-test.ipynb: The significance test

Among all scripts, 2-1, 2-2, 3-1 are forked most from the original EC^2-VAE and Music InpaintNet repos. 

## Credit
Please cite this paper if you want to use this work or base this work for the further project:
> @inproceedings{music-sketchnet, <br>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; author = {Ke Chen and Cheng-i Wang and Taylor Berg-Kirkpatrick and Shlomo Dubnov}, <br>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; title = {Music SketchNet: Controllable Music Generation via Factorized Representations of Pitch and Rhythm}, <br>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; booktitle = {Proceedings of the 21th International Society for Music Information Retrieval Conference, {ISMIR}}, <br>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; year = {2020}<br>
> }
