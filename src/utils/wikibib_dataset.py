import torch, os, sys, random
from tqdm import tqdm
import numpy as np

class WikiBibDataset(torch.utils.data.Dataset):
    def __init__(self,
                 tokenizer,
                 source_directory,
                 target_directory,
                 train_sampler,
                 max_len,
                 source_lang=None,
                 target_lang=None,
                 num_langs=0,
                 file_path=None,
                 langs_to_use=None,
                 id2lang=None,
                 remove_underscore=False,
                 sampling_factor=1.0,
                 seed=42):
        self.seed = seed
        self.remove_underscore = remove_underscore
        self.langs_to_use = langs_to_use
        self.id2lang = id2lang

        self.num_langs = num_langs

        self.tokenizer = tokenizer
        self.max_len = max_len
        self.source_directory = source_directory
        self.target_directory = target_directory

        self.source_lang = source_lang
        self.target_lang = target_lang

        self.train_sampler = train_sampler

        if file_path:
            self.source_examples = self.read_file(file_path)
        else:
            self.source_examples = self.read_files(source_directory, source=True)
        self.target_examples = self.read_files(target_directory)


        print('Dataset created, with settings: source: {}\n target: {} \n max len: {} \n source_lang: {}\n target_lang: {}'.format(self.source_directory, self.target_directory, self.max_len, self.source_lang, self.target_lang))

        self.finalized_examples = None

        if self.train_sampler == 'baseline':
            self.finalized_examples = self.create_baseline_dataset()
            print('Baseline sampling: created {} examples ready for training.'.format(len(self.finalized_examples)))

        elif self.train_sampler == 'upsample':
            self.finalized_examples = self.create_upsampled_dataset()
            print('Upsampled target dataset.')
        
        elif self.train_sampler == 'weighted':
            self.sampling_factor = sampling_factor
            assert self.sampling_factor >= 0 
            self.source_examples, self.sizes = self.read_files_weight_sampling(source_directory, source=True)
            self.finalized_examples = self.create_weighted_sampling_dataset()

        print('First example: {}'.format(self.finalized_examples[0]))

    def __getitem__(self, idx):
        instance = self.finalized_examples[idx]

        enc = self.tokenizer(
            instance,
            max_length=self.max_len,
            truncation=True,
            return_token_type_ids= False,
            return_tensors='pt'
        )

        return {
            'input_ids': enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
        }

    def __len__(self):
        return len(self.finalized_examples)

    def read_files(self, directory, source=False):

        if directory is None:
            return []

        inps = []
        files = [directory + x for x in os.listdir(directory)]
        files.sort()

        with tqdm(total=len(files), file=sys.stdout, ascii=True) as pbar:
            for i, file in enumerate(files):
                # if i >= self.num_langs:
                #     print(f"Skipping lang {i} due to num langs set to {self.num_langs}")
                #     continue
                if self.id2lang and (not self.id2lang[i] in self.langs_to_use):
                    print(f"Skipping lang {i} ({self.id2lang[i]}) due to not in {self.langs_to_use}")
                    continue
 
                with open(file, 'r') as f:
                    for line in f:
                        if self.remove_underscore:
                            temp = ""
                            for token in self.tokenizer.tokenize(line.strip()):
                                if token != "▁":
                                    temp += token.replace("▁", " ")
                            inps.append(temp)
                        else:
                            inps.append(line.strip())
                print('Dataset: Finished loading file: {}'.format(file))
                pbar.update(1)

        if source:
            print('Read in {} source examples across {} files'.format(len(inps), len(files)))
        else:
            print('Read in {} target examples across {} files'.format(len(inps), len(files)))

        return inps

    def read_files_weight_sampling(self, directory, source=False):

        if directory is None:
            return []

        inps = []
        sizes = []
        files = [directory + x for x in os.listdir(directory)]
        files.sort()

        with tqdm(total=len(files), file=sys.stdout, ascii=True) as pbar:
            for i, file in enumerate(files):
                # if i >= self.num_langs:
                #     print(f"Skipping lang {i} due to num langs set to {self.num_langs}")
                #     continue
                if self.id2lang and (not self.id2lang[i] in self.langs_to_use):
                    print(f"Skipping lang {i} ({self.id2lang[i]}) due to not in {self.langs_to_use}")
                    continue
                inp = []
                with open(file, 'r') as f:
                    for line in f:
                        if self.remove_underscore:
                            temp = ""
                            for token in self.tokenizer.tokenize(line.strip()):
                                if token != "▁":
                                    temp += token.replace("▁", " ")
                            inp.append(temp)
                        else:
                            inp.append(line.strip())
                print('Dataset: Finished loading file: {}'.format(file))
                pbar.update(1)
                inps.append(inp) # each inp is a list of sequences
                sizes.append(len(inp)) # contains the number of sequences in each file (lang)

        if source:
            print('Read in {} source examples across {} files'.format(sum(sizes), len(files)))
        else:
            print('Read in {} target examples across {} files'.format(sum(sizes), len(files)))

        return inps, sizes

    def create_baseline_dataset(self):
        x = self.source_examples + self.target_examples
        random.seed(self.seed)
        random.shuffle(x)
        return x

    def read_file(self, file_path, source=False):

        inps = []
        with open(file_path, 'r') as f:
            for line in f:
                if self.remove_underscore:
                    temp = ""
                    for token in self.tokenizer.tokenize(line.strip()):
                        if token != "▁":
                            temp += token.replace("▁", " ")
                    inps.append(temp)

                else:
                    inps.append(line.strip())
 
        print('Dataset: Finished loading file: {}'.format(file_path))
        print('Read in {} examples across'.format(len(inps)))

        return inps

    def create_upsampled_dataset(self):

        num_source_examples = len(self.source_examples)

        random.seed(self.seed)
        target_samples = random.choices(self.target_examples, k = num_source_examples)

        print('Starting with {} source examples and {} target examples'.format(num_source_examples, len(self.target_examples)))
        print('Sampled {} examples from target'.format(len(target_samples)))

        x = self.source_examples + target_samples
        random.shuffle(x)

        print('Created {} examples for training'.format(len(x)))

        return x
    
    def create_weighted_sampling_dataset(self):
        probs = np.array([1.0 * size for size in self.sizes])
        probs /= probs.sum()
        probs = np.array([p ** self.sampling_factor for p in probs])
        probs /= probs.sum()

        assert probs.sum() == 1.0

        random.seed(self.seed)
        np.random.seed(self.seed)
        lang_counter = [0] * self.num_langs # keep count how many langs we have sampled from for each lang
        x = []
        total_size = sum(self.sizes) # total number of examples in all langs
        sample_lang_ids = np.random.choice(self.num_langs, total_size, p=probs) # sample lang_id from all examples in all langs
        for lang_id in sample_lang_ids:
            if lang_counter[lang_id] < self.sizes[lang_id]:
                # if we have not sampled all examples from this lang, sample one example from this lang
                x.append(self.source_examples[lang_id][lang_counter[lang_id]])
                lang_counter[lang_id] += 1
            else:
                # if we have sampled all examples from this lang, sample one example (repeatedly) from this langs
                rand_idx = random.randint(0, (self.sizes[lang_id]) - 1)
                x.append(self.source_examples[lang_id][rand_idx])
        print('#############lang_counter#############')
        print(lang_counter)
        random.shuffle(x)
        return x
