import argparse

from oplas.models import Projector

parser = argparse.ArgumentParser(prog='count_params')

parser.add_argument('--inner', default='8', type=int)
parser.add_argument('--hidden', default='8', type=int)
parser.add_argument('--proj', default='64', type=int)
args = parser.parse_args()

inner = args.inner
hidden = args.hidden
proj = args.proj

# projector = Projector(64,64,num_inner_layers=6,hidden_dims_scale=16)
# projector = Projector(64,64,num_inner_layers=4,hidden_dims_scale=12)
projector = Projector(64, proj, num_inner_layers=inner, hidden_dims_scale=hidden)

# Count all parameters
total_params = sum(p.numel() for p in projector.parameters())

# Count only trainable parameters
trainable_params = sum(p.numel() for p in projector.parameters() if p.requires_grad)

print(f"Inner layers: {inner}, hidden scale: {hidden}, projector dims: {proj}")
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
