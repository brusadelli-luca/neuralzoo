import numpy as np
import matplotlib.pyplot as plt

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def filtre(dataset, filter_labels):
    labels_filt = []
    data_filt = []
    filenames_filt = []

    for i in range( len(dataset[b'labels']) ):
        if dataset[b'labels'][i] in filter_labels:
            labels_filt.append(dataset[b'labels'][i])
            data_filt.append(dataset[b'data'][i].tolist())
            filenames_filt.append(dataset[b'filenames'][i])
    
    return {b'batch_label': b'full filtered training dataset', b'labels': labels_filt, b'data': data_filt, b'filenames': filenames_filt}

def info(dataset):
    
    print('dataset label :', dataset[b'batch_label'])
    print('labels len :', len(dataset[b'labels']))
    print('data len :', len(dataset[b'data']))
    print('filenames len :', len(dataset[b'filenames']))

def display_img(dataset, meta, index):

    img = np.reshape(dataset[b'data'][index], (3, 1024))
    R = img[0]
    G = img[1]
    B = img[2]
    
    RGB = []

    for i in range(1024):
        RGB.append((R[i], G[i], B[i]))
        
    RGB = np.reshape(RGB, (32, 32, 3))
    
    plt.imshow(RGB)
    plt.title(str(dataset[b'filenames'][index]) + '\n' + 'label : ' + str(meta[b'label_names'][dataset[b'labels'][index]]))
    plt.axis('off')


def make_autopct(values):
    def my_autopct(pct):
        total = sum(values)
        val = int(round(pct*total/100.0))
        return '{p:.0f} %\n({v:d})'.format(p=pct,v=val)
    return my_autopct