# STEPS

<!-- DOWNLOAD STABLE AUDIO 1.0 via hugging face  -->

huggingface-cli login

huggingface-cli download stabilityai/stable-audio-open-1.0

By default it will save it in .cache directory :
~/.cache/huggingface/hub/models--stabilityai--stable-audio-open-1.0)

you can copy it locally to the folder -> mv ~/.cache/huggingface/hub/models--stabilityai--stable-audio-open-1.0/snapshots/<some_id> ../yourpath/stable-audio-open-1.0/

I will be leaving this directory empty

Else you can do it through git lfs :
git lfs install
git clone https://huggingface.co/stabilityai/stable-audio-open-1.0

<!-- Install Packages -->

pip install -r requirements.txt

<!-- Run the notebook -->
