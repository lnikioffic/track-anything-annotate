import torch

if torch.cuda.is_available():
    print('Using GPU')
    DEVICE = 'cuda'
else:
    print('CUDA not available. Please connect to a GPU instance if possible.')
    DEVICE = 'cpu'


XMEM_CONFIG = {
    'top_k': 50,
    'mem_every': 5,
    'deep_update_every': -1,
    'enable_long_term': True,
    'enable_long_term_count_usage': True,
    'num_prototypes': 128,
    'min_mid_term_frames': 10,
    'max_mid_term_frames': 15,
    'max_long_term_elements': 5000,
    'size': 480,
    'device': DEVICE,
}
