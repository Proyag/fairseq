#!/usr/bin/env python3

import argparse
import torch


def get_parser():
    parser = argparse.ArgumentParser(
        description="applies trained masks on Linear layers"
    )
    # fmt: off
    parser.add_argument('-i', '--checkpoint-in', required=True, help='checkpoint file to read')
    parser.add_argument('-o', '--checkpoint-out', required=True, help='checkpoint file to write to')
    parser.add_argument('-c', '--check-original', required=False, help='pre-finetuning checkpoint to verify frozen parameters')
    parser.add_argument('-C', '--check-only', required=False, action='store_true', help='only check')
    # fmt: on

    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint_in)

    if args.check_original:
        orig_model = torch.load(args.check_original)['model']

    # Set this config entry to False to disable masking during decoding with this checkpoint
    # since the mask is already applied here
    checkpoint['cfg'].masking.masked_finetune = False
    threshold = checkpoint['cfg'].masking.masked_finetune_threshold

    for k in list(checkpoint['model'].keys()):
        if args.check_original and not k.endswith('.mask'):
            if torch.eq(checkpoint['model'][k], orig_model[k]).all():
                print("Equal: {}".format(k))
            else:
                print("NOT equal {}".format(k))

        elif k.endswith('.mask') and not args.check_only:
            layer_name = '.'.join(k.split('.')[:-1])
            weight = layer_name + '.weight'
            mask = k

            # Check shapes match
            assert checkpoint['model'][k].shape == checkpoint['model'][weight].shape, "Shape mismatch between weight matrix and corresponding mask"

            # Apply mask
            checkpoint['model'][weight] = torch.ge(checkpoint['model'][mask], threshold) * checkpoint['model'][weight]

            # Remove mask from checkpoint
            del checkpoint['model'][mask]

    if not args.check_only:
        torch.save(checkpoint, args.checkpoint_out)


if __name__ == "__main__":
    main()
