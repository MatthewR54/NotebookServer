import torch

# The original UniversalFakeDetect repository saves the state dict with underscores (e.g., "fc_weight") rather
# than the default period delimiter (e.g., "fc.weight"). This simple script allows us to save the
# trained weights from the original UFD repository and make them compatible with the new repo.
orig = torch.load("C:\\Users\\Janus\\Documents\\NotebookServer\\github_model_srcs\\pretrained_weights\\ufd_weights.pth")

modified = {}
for key, val in orig.items():
    new_key = key.replace("_", ".")
    modified[new_key] = val

torch.save(modified, 'C:\Users\Janus\Documents\NotebookServer\github_model_srcs\pretrained_weights\ufd_weights_FIX.pth')