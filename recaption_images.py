#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File name: recaption_images.py
# Description: Recaptioning images with automatically generated captions.
# Example Usage: python3 recaption_images.py -i /home/feng/Downloads/images -o /home/feng/Downloads/recaptioned_images
# Author: Feng Wang
# Date: 2021-11-01
# Dependency: [torch, torchvision, numpy, imageio, tqdm, pillow, glob]
# License: AGPL-v3
#

import torch
import torch.nn.functional as F
import numpy as np
import json
import torchvision.transforms as transforms
from PIL import Image
from imageio import imread
import os
import sys
import glob
import tqdm
import shutil

def imresize( array, resolution ):
    return np.array( Image.fromarray(array).resize( resolution ) )

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def caption_image_beam_search(encoder, decoder, image_path, word_map, beam_size=3):

    k = beam_size
    vocab_size = len(word_map)

    # Read image and process
    img = imread(image_path)
    if len(img.shape) == 2:
        img = img[:, :, np.newaxis]
        img = np.concatenate([img, img, img], axis=2)
    img = imresize(img, (256, 256))
    img = img.transpose(2, 0, 1)
    img = img / 255.
    img = torch.FloatTensor(img).to(device)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([normalize])
    image = transform(img)  # (3, 256, 256)

    # Encode
    image = image.unsqueeze(0)  # (1, 3, 256, 256)
    encoder_out = encoder(image)  # (1, enc_image_size, enc_image_size, encoder_dim)
    enc_image_size = encoder_out.size(1)
    encoder_dim = encoder_out.size(3)

    # Flatten encoding
    encoder_out = encoder_out.view(1, -1, encoder_dim)  # (1, num_pixels, encoder_dim)
    num_pixels = encoder_out.size(1)

    # We'll treat the problem as having a batch size of k
    encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)  # (k, num_pixels, encoder_dim)

    # Tensor to store top k previous words at each step; now they're just <start>
    k_prev_words = torch.LongTensor([[word_map['<start>']]] * k).to(device)  # (k, 1)

    # Tensor to store top k sequences; now they're just <start>
    seqs = k_prev_words  # (k, 1)

    # Tensor to store top k sequences' scores; now they're just 0
    top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)

    # Tensor to store top k sequences' alphas; now they're just 1s
    seqs_alpha = torch.ones(k, 1, enc_image_size, enc_image_size).to(device)  # (k, 1, enc_image_size, enc_image_size)

    # Lists to store completed sequences, their alphas and scores
    complete_seqs = list()
    complete_seqs_alpha = list()
    complete_seqs_scores = list()

    # Start decoding
    step = 1
    h, c = decoder.init_hidden_state(encoder_out)

    # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
    while True:

        embeddings = decoder.embedding(k_prev_words).squeeze(1)  # (s, embed_dim)

        awe, alpha = decoder.attention(encoder_out, h)  # (s, encoder_dim), (s, num_pixels)

        alpha = alpha.view(-1, enc_image_size, enc_image_size)  # (s, enc_image_size, enc_image_size)

        gate = decoder.sigmoid(decoder.f_beta(h))  # gating scalar, (s, encoder_dim)
        awe = gate * awe

        h, c = decoder.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))  # (s, decoder_dim)

        scores = decoder.fc(h)  # (s, vocab_size)
        scores = F.log_softmax(scores, dim=1)

        # Add
        scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)

        # For the first step, all k points will have the same scores (since same k previous words, h, c)
        if step == 1:
            top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
        else:
            # Unroll and find top scores, and their unrolled indices
            top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)

        # Convert unrolled indices to actual indices of scores
        prev_word_inds = top_k_words / vocab_size  # (s)
        next_word_inds = top_k_words % vocab_size  # (s)

        # Add new words to sequences, alphas
        #seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)
        seqs = torch.cat([seqs[prev_word_inds.long()], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)
        #seqs_alpha = torch.cat([seqs_alpha[prev_word_inds], alpha[prev_word_inds].unsqueeze(1)],
        #                       dim=1)  # (s, step+1, enc_image_size, enc_image_size)
        seqs_alpha = torch.cat([seqs_alpha[prev_word_inds.long()], alpha[prev_word_inds.long()].unsqueeze(1)],
                               dim=1)  # (s, step+1, enc_image_size, enc_image_size)

        # Which sequences are incomplete (didn't reach <end>)?
        incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                           next_word != word_map['<end>']]
        complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

        # Set aside complete sequences
        if len(complete_inds) > 0:
            complete_seqs.extend(seqs[complete_inds].tolist())
            complete_seqs_alpha.extend(seqs_alpha[complete_inds].tolist())
            complete_seqs_scores.extend(top_k_scores[complete_inds])
        k -= len(complete_inds)  # reduce beam length accordingly

        # Proceed with incomplete sequences
        if k == 0:
            break
        seqs = seqs[incomplete_inds]
        seqs_alpha = seqs_alpha[incomplete_inds]
        h = h[prev_word_inds[incomplete_inds].long()]
        c = c[prev_word_inds[incomplete_inds].long()]
        encoder_out = encoder_out[prev_word_inds[incomplete_inds].long()]
        top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
        k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

        # Break if things have been going on too long
        if step > 50:
            break
        step += 1

    i = complete_seqs_scores.index(max(complete_seqs_scores))
    seq = complete_seqs[i]
    alphas = complete_seqs_alpha[i]

    return seq, alphas




checkpoint = None
rev_word_map = None
word_map = None

# caption_it( '/home/feng/Downloads/t0172e549c7b3337a63.png', '/home/feng/Downloads/BEST_checkpoint_coco_5_cap_per_img_5_min_word_freq.pth.tar', '/home/feng/Downloads/WORDMAP_coco_5_cap_per_img_5_min_word_freq.json' )
def caption_it( img, model_path, word_map_path, beam_size=5 ):

    global checkpoint
    if checkpoint is None:
        checkpoint = torch.load(model_path, map_location=str(device))

    decoder = checkpoint['decoder']
    decoder = decoder.to(device)
    decoder.eval()
    encoder = checkpoint['encoder']
    encoder = encoder.to(device)
    encoder.eval()

    global rev_word_map
    global word_map
    if rev_word_map is None or word_map is None:
        with open(word_map_path, 'r') as j:
            word_map = json.load(j)
        rev_word_map = {v: k for k, v in word_map.items()}  # ix2word

    # Encode, decode with attention and beam search
    seq, alphas = caption_image_beam_search(encoder, decoder, img, word_map, beam_size)
    #alphas = torch.FloatTensor(alphas)

    words = [ rev_word_map[ind] for ind in seq ]
    w = ' '.join( words[1:-1] )
    return w


def download_remote_model(model_path, model_url):
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    file_name = model_url.rsplit('/', 1)[1]
    local_model_path = os.path.join( model_path, file_name )
    if not os.path.isfile(local_model_path):
        print( f'downloading model file {local_model_path} from {model_url}' )
        with open(local_model_path, "wb") as file:
            response = get(model_url)
            file.write(response.content)
        print( f'downloaded model file {local_model_path} from {model_url}' )

    return local_model_path


def recaption_images( input_folder, output_folder=None ):

    # the output folder
    if output_folder is None:
        output_folder = f'{input_folder}_with_caption'
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

    # the model folder
    user_model_path = os.path.join( os.path.expanduser('~'), '.deepoffice', 'recaption_images', 'model' )

    # download model files
    model_path = download_remote_model( user_model_path, 'TODO' )
    word_map_path = download_remote_model( user_model_path, 'TODO' )

    # process images
    images = glob.glob( f'{input_folder}/*.png' ) + glob.glob( f'{input_folder}/*.jpg' ) + glob.glob( f'{input_folder}/*.jpeg' ) + glob.glob( f'{input_folder}/*.bmp' )
    for image in tqdm.tqdm(images):
        # generate caption
        caption = caption_it( image, model_path, word_map_path )
        old_image_name = model_url.rsplit('/', 1)[1]
        # generate new file name
        new_image_path = f'{output_folder}/{caption}_{old_image_name}'
        # copy file
        shutil.copyfile(image, new_image_path)
        print( f'rewriting {image} as {new_image_path}' )

import argparse

def get_arguments():
    parser = argparse.ArgumentParser(description='Recaption Images')
    parser.add_argument('-i', '--input', type=str, help='The path to the input images.')
    parser.add_argument('-o', '--output', type=str, help='The path to the recaptioned images.')
    args = parser.parse_args()
    return args.input, args.output


if __name__ == '__main__':
    recaption_images( *get_arguments() )

