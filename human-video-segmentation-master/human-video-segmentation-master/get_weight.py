from hvsnet import HVSNet

import torch
import argparse
import os

if __name__ == '__main__':
    if not os.path.exists('saved/models'):
        os.mkdir('saved/models')

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, help='path of checkpoint')

    args = parser.parse_args()
    assert(args.checkpoint.endswith('pth'))

    file = os.path.basename(args.checkpoint)
    path = os.path.join('saved/models', file.replace('pth', 'pt'))

    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    torch.save(checkpoint['model'], path)
