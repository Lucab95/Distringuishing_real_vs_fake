# Code for "TSM: Temporal Shift Module for Efficient Video Understanding"
# arXiv:1811.08383
# Ji Lin*, Chuang Gan, Song Han
# {jilin, songhan}@mit.edu, ganchuang@csail.mit.edu
# ------------------------------------------------------
# Code adapted from https://github.com/metalbubble/TRN-pytorch/blob/master/process_dataset.py

from encodings import utf_8
import os


# dataset_path = 'video/kinetics'
dataset_path = 'video/sasefe'
label_path = 'video'
# label_path = r'C:\Users\luca9\Desktop\kinetics-dataset-main'

if __name__ == '__main__':
    with open('tools/sasefe_label_map.txt') as f:
        categories = f.readlines()
        # print("categ", categories)
        categories = [c.strip().replace(' ', '_').replace('"', '').replace('(', '').replace(')', '').replace("'", '') for c in categories]
    # assert len(set(categories)) == 400
    dict_categories = {}
    for i, category in enumerate(categories):
        dict_categories[category] = i


    files_input = ['train.csv','val.csv',]
    # files_input = ['train.csv']
    files_output = ['train_videofolder.txt','val_videofolder.txt']
    # files_output = ['train_videofolder.txt',]
    for (filename_input, filename_output) in zip(files_input, files_output):
        count_cat = {k: 0 for k in dict_categories.keys()}
        print(dict_categories.keys())
        with open(os.path.join(label_path, filename_input)) as f: # path to tge folder dn2sur/age, dn2sur/Ahmed and so on
            lines = f.readlines()[1:]
        folders = []
        idx_categories = []
        categories_list = []
        for line in lines:
            line = line.rstrip() #remove the \n
            items = line.split(',')
            folders.append(items[1] + '_' + items[2])
            this_catergory = items[0].replace(' ', '_').replace('"', '').replace('(', '').replace(')', '').replace("'", '')
            print(items[1] + '_' + items[2])
            print(this_catergory)
            categories_list.append(this_catergory)
            idx_categories.append(dict_categories[this_catergory])
            count_cat[this_catergory] += 1
        print(max(count_cat.values()))
        # assert len(idx_categories) == len(folders)
        missing_folders = []
        output = []
        for i in range(len(folders)):
            curFolder = folders[i]
            curIDX = idx_categories[i]
            # counting the number of frames in each video folders
            img_dir = os.path.join(dataset_path, categories_list[i], curFolder)
            if not os.path.exists(img_dir):
                missing_folders.append(img_dir)
                # print(missing_folders)
            else:
                dir_files = os.listdir(img_dir)
                output.append('%s %d %d'%(os.path.join(categories_list[i], curFolder), len(dir_files), curIDX))
            print('%d/%d, missing %d'%(i, len(folders), len(missing_folders)))
        with open(os.path.join(label_path, filename_output),'w') as f:
            f.write('\n'.join(output))
        with open(os.path.join(label_path, 'missing_' + filename_output),'w') as f:
            f.write('\n'.join(missing_folders))
