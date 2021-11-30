import pickle
import numpy as np

def gen_mmit_graph_adj(category_file, anno_file, class_num, output_adj_file):

    occur_num_per_class = []
    occur_num_two_classes = []

    for i in range(class_num):
        occur_num_per_class.append(0)
        occur_num_two_classes.append([0] * class_num)

    with open(anno_file, "r") as f:
        for line in f:
            line = line.strip("\n").strip()
            labels = line.split(',', 1)[-1].strip()
            labels = labels.split(',')
            for i in labels:
                occur_num_per_class[int(i)] += 1
                for j in labels:
                    if i != j:
                        occur_num_two_classes[int(i)][int(j)] += 1
    
    occur_num = np.array(occur_num_per_class)
    graph_adj = np.array(occur_num_two_classes)
    
    occur_num = np.expand_dims(occur_num, axis=1)
    graph_adj = graph_adj / occur_num
    
    with open(output_adj_file, 'wb') as f:
        pickle.dump(graph_adj, f, protocol=4)

if __name__ == '__main__':
    category_file = "moments_categories.txt"
    anno_file = "trainingSet.txt"
    class_num = 313
    output_adj_file = "mmit_adj.pkl"
    gen_mmit_graph_adj(category_file, anno_file, class_num, output_adj_file)
