# Source : https://github.com/GuessWhatGame/vqa

import io
import json
import collections
import argparse

from data_provider.vqa_dataset import VQADataset
from nltk.tokenize import TweetTokenizer
from distutils.util import strtobool

if __name__ == '__main__':


    parser = argparse.ArgumentParser('Creating dictionary..')

    parser.add_argument("-data_dir", type=str, help="Path to VQA dataset")
    parser.add_argument("-dict_file", type=str, default="dict.json", help="Name of the dictionary file")
    parser.add_argument("-year", type=int, default=2014, help="VQA dataset year (2014/2017)")
    parser.add_argument("-min_occ", type=int, default=2,
                        help='Minimum number of occurences to add word to dictionary')
    parser.add_argument("-ans_to_keep", type=int, default=2000, help='Number of answers to keep')
    parser.add_argument("-ans_preprocess", type=lambda x:bool(strtobool(x)), default="False", help='preprocess answers (higher accuracy but slow start)')
    parser.add_argument("-merge_val", type=lambda x:bool(strtobool(x)), default="False", help='Fuse train/val dataset')

    args = parser.parse_args()



    print("Loading dataset...")
    train_dataset = VQADataset(args.data_dir, args.year, "train", preprocess_answers=args.ans_preprocess)
    answer_counters = train_dataset.answer_counter.most_common()
    games = train_dataset.games

    if args.merge_val:
        valid_dataset = VQADataset(args.data_dir, args.year, "val", preprocess_answers=args.ans_preprocess)
        answer_counters += valid_dataset.answer_counter.most_common()
        games += valid_dataset.games

    word2i = {'<unk>': 0,
              '<start>': 1,
              '<stop>': 2,
              '<padding>': 3
              }

    answer2i = {'<unk>': 0}

    word2occ = collections.defaultdict(int)
    answer2occ = collections.Counter()

    print("Creating dictionary...")
    for k, v in answer_counters:
        answer2occ[k] += v

    selected = sum([v[1] for v in answer2occ.most_common(args.ans_to_keep)])
    total = sum([v[1] for v in answer2occ.most_common()])


    # Input words
    tknzr = TweetTokenizer(preserve_case=False)

    for game in games:
        input_tokens = tknzr.tokenize(game.question)
        for tok in input_tokens:
            word2occ[tok] += 1


    included_cnt = 0
    excluded_cnt = 0
    for word, occ in word2occ.items():
        if occ >= args.min_occ and word.count('.') <= 1:
            included_cnt += occ
            word2i[word] = len(word2i)
        else:
            excluded_cnt += occ


    for i, answer in enumerate(answer2occ.most_common(args.ans_to_keep)):
        answer2i[answer[0]] = len(answer2i)


    print("Number of words (occ >= {0:}): {1:} ~ {2:.2f}%".format(args.min_occ, len(word2i), 100.0*len(word2i)/len(word2occ)))
    print("Number of answers: {0:}".format(len(answer2i)))


    print("Dumping file...")
    with io.open(args.dict_file, 'wb') as f_out:
       data = json.dumps({'word2i': word2i, 'answer2i': answer2i, "preprocess_answers": args.ans_preprocess})
       f_out.write(data.encode('utf8', 'replace'))

    print("Done!")

