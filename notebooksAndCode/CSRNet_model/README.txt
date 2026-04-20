Project structure:

├── model.py              # CSRNet architecture
├── train.py              # training + validation loop
├── image.py              # data loader (image + density map)
├── dataset.py            # dataset class
├── utils.py              # checkpoint saving
├── split_data.py		   # Split the Data to training/validation/test
├── train.json            # training image paths
├── val.json              # validation image paths
├── test.json             # test image paths
│
├── weights.pth           # pretrained CSRNet weights
│
├── training_data/
│   ├── unlabeled_frames_02_05/
│   ├── ground_truth_02_05/
│   ├── unlabeled_frames_03_10/        # optional additional data
│   ├── ground_truth_03_10/


Create Environment:
conda create -n csrnet python=3.9
conda activate csrnet
pip install torch torchvision
pip install numpy==1.26.4
pip install matplotlib
pip install h5py
pip install opencv-python

Data Format:
Images/frames from the game recording is saved as: training_data/unlabeled_frames_xx_xx/frame_XXXX.png
Annotation data is saved as: training_data/annotations_xx_xx/frame_XXXX.npy
Training Density Map is saved as: training_data/ground_truth_xx_xx/frame_XXXX.h5
Where xx_xx is the date of the recording, and XXXX is the frame index
To access the original recording please visit: https://webcams.mtu.edu/broomball/black/

Run the training:
python train.py train.json val.json --pre weights.pth 0 csrnet_mtu (using pre-trained weights)
The training result will be saved as: csrnet_mtu_model_best.pth.tar

Evaluation:
To evaluate the results from the report please type: 
checkpoint = torch.load("Training Results and Checkpoints\csrnet_mtu_run1model_best.pth.tar")
model.load_state_dict(checkpoint['state_dict'])