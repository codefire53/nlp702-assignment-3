from glob import glob
from tqdm import tqdm
import random
import json

def load_source_files(split, source_files, all_texts_dict, lang_map):
    for source_file in tqdm(source_files):
        with open(source_file, 'r') as f:
            lines = f.readlines()
            for idx, line in tqdm(enumerate(lines)):
                if idx == 0 :
                    continue
                sent_id, sent_split, lang, sent = line.split('\t', 3)
                if split in sent_split and sent_id in all_texts_dict:
                    all_texts_dict[sent_id]['sources'].append(sent)
                    all_texts_dict[sent_id]['dialects'].append(lang_map[lang])
    
    # filtered out any id that doesnt have parallel sentences and expand the rows otherwise
    all_keys = [key for key in all_texts_dict.keys()]
    for key_id in tqdm(all_keys):
        if len(all_texts_dict[key_id]['sources']) == 0:
            del all_texts_dict[key_id]
        else:
            tmp_sources = all_texts_dict[key_id]['sources'].copy()
            target = all_texts_dict[key_id]['target']
            tmp_dialects = all_texts_dict[key_id]['dialects'].copy()
            del all_texts_dict[key_id]
            for idx, source in enumerate(tmp_sources):
                new_key = f"{key_id}_{idx}#{tmp_dialects[idx]}"
                
                all_texts_dict[new_key] = {
                    'source': source,
                    'target': target
                }

            
def load_target_file(split, target_file, all_texts_dict, lang_map):
    with open(target_file, 'r') as f:
        lines = f.readlines()
        for idx, line in tqdm(enumerate(lines)):
            if idx == 0 :
                continue
            sent_id, sent_split, lang_id, sent = line.split('\t', 3)
            if split in sent_split:
                all_texts_dict[sent_id] = {
                    'sources': [],
                    'dialects': [],
                    'target': sent
                }


def train_val_split(all_texts_dict, val_split):
    random.seed(42)
    all_keys = [key for key in all_texts_dict.keys()]
    sz = len(all_keys)
    val_sz = int(sz*val_split)
    val_keys = random.sample(all_keys, val_sz)
    train_keys = [key for key in all_keys if key not in val_keys]
    train_lst = []
    train_dialects = []
    val_lst = []
    val_dialects = []
    for key in tqdm(all_keys):
        if key in val_keys:
            val_lst.append((all_texts_dict[key]['source'].strip(), all_texts_dict[key]['target'].strip()))
            val_dialect = key.rsplit('#', 1)
            val_dialect = val_dialect[-1]
            val_dialects.append(val_dialect)
        else:
            train_lst.append((all_texts_dict[key]['source'].strip(), all_texts_dict[key]['target'].strip())) 
            train_dialect = key.rsplit('#', 1)
            train_dialect = train_dialect[-1]
            train_dialects.append(train_dialect)
    
    return train_lst, val_lst, train_dialects, val_dialects

def write_to_file(filename, lst, dialects=[]):
    if len(set(dialects)) > 0:
        dialect_lst = list(set(dialects))
        for dialect in tqdm(dialect_lst):
            prefix, suffix = filename.rsplit('.', 1)
            dialect_filename = f"{prefix}_{dialect}.{suffix}"
            with open(dialect_filename, 'w') as f:
                for idx, row in tqdm(enumerate(lst)):
                    if dialect == dialects[idx]:
                        f.write('\t'.join(row)+'\n')
    else:
        with open(filename, 'w') as f:
            for idx, row in tqdm(enumerate(lst)):
                f.write('\t'.join(row)+'\n')



def create_corpuses(split, source_files, target_file, train_filename, val_filename, lang_map, val_split=0.2):
    all_texts_dict = dict()
    load_target_file(split, target_file, all_texts_dict, lang_map)
    load_source_files(split, source_files, all_texts_dict, lang_map)
    train_lst, val_lst, train_dia, val_dia = train_val_split(all_texts_dict, val_split)
    write_to_file(train_filename, train_lst, train_dia)
    write_to_file(train_filename, train_lst)
    write_to_file(val_filename, val_lst)
    write_to_file(val_filename, val_lst, val_dia)




if __name__ == '__main__':
    train_filename = './train_madar.tsv'
    val_filename = './val_madar.tsv'
    lang_map_file = './lang_map.json'
    with open(lang_map_file, 'r') as f:
        lang_map = json.load(f)
    all_files = [name for name in glob('./*.tsv')]
    source_files = [name for name in all_files if 'MSA' not in name]
    target_file = ''
    for name in all_files:
        if 'MSA' in name:
            target_file = name
            break
    split = 'corpus-26-train'
    create_corpuses(split, source_files, target_file, train_filename, val_filename, lang_map, 0.2)    

