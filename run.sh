#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=2 python neural_style.py --content  'examples/1-content.jpg' --styles 'examples/1-style1.jpg' 'examples/1-style2.jpg' --output 'examples/1-output.jpg' --iterations 150

CUDA_VISIBLE_DEVICES=2 python neural_style.py --content  'examples/2-content.jpg' --styles 'examples/2-style1.jpg' 'examples/2-style2.jpg' --output 'examples/2-output.jpg' --iterations 150
