import pickle
import numpy as np
import torchtext.vocab as vocab

def gen_mmit_text_lexicon(category_file, class_num, output_lexicon_file):
    glove = vocab.GloVe(name="6B", dim=300)
    total_text_vec = np.array([])

    with open(category_file, "r") as f:
        for line in f:
            line = line.strip("\n").strip()
            category_words = line.split(',', 1)[0].strip()
            
            if '+' in category_words:
                category_words = category_words.split('+')
            elif '/' in category_words:
                category_words = category_words.split('/')
            elif ' ' in category_words:
                category_words = category_words.split(' ')
            
            if type(category_words) is list and len(category_words) > 0:
                text_vec_list = []
                for word in category_words:
                    text_vec_list.append(glove.vectors[glove.stoi[word]])
                text_vec = sum(text_vec_list) / len(text_vec_list)
            else:
                text_vec = glove.vectors[glove.stoi[category_words]]
            total_text_vec = np.append(total_text_vec, text_vec.numpy())
    
    total_text_vec = total_text_vec.reshape(class_num, -1)
    with open(output_lexicon_file, 'wb') as f:
        pickle.dump(total_text_vec, f, protocol=4)

if __name__ == '__main__':
    category_file = "moments_categories.txt"
    class_num = 313
    output_lexicon_file = "text_lexicon.pkl"
    gen_mmit_text_lexicon(category_file, class_num, output_lexicon_file)
