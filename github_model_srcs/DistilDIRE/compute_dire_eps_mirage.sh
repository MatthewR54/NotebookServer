## set MODEL_PATH, num_samples, has_subfolder, images_dir, recons_dir, dire_dir
DATA_ROOT=("C:\Users\Janus\Documents\NotebookServer\github_model_srcs\mirage_data")
SAVE_ROOT=("C:\Users\Janus\Documents\NotebookServer\github_model_srcs\mirage_data_dire")
USE_LIBUV=0

MODEL_PATH="C:\Users\Janus\Documents\NotebookServer\github_model_srcs\pretrained_weights\256x256-adm.pt" # imagenet pretrained adm (unconditional, 256x256)
SAMPLE_FLAGS="--batch_size 16" # ddim20 is forced
PREPROCESS_FLAGS="--compute_dire True --compute_eps True"

SAVE_FLAGS="--data_root ${DATA_ROOT} --save_root ${SAVE_ROOT}"
echo "Running on ${DATA_ROOT} with save root ${SAVE_ROOT}"
torchrun ./guided_diffusion/compute_dire_eps.py --model_path $MODEL_PATH $PREPROCESS_FLAGS $SAMPLE_FLAGS $SAVE_FLAGS
