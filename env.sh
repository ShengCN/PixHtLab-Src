# conda create -n pixht python=3.9  -y
# conda activate pixht

conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch -y

cd Docker/imagen && pip install . && cd - 
cd Demo/Torch_Render && pip install . && cd -

pip install matplotlib pandas scipy dominate tensorboard opencv-python h5py 
pip install holoviews==1.17.0
pip install --force-reinstall --user param==1.13.0 numpy==1.26.0
