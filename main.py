import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from keras.models import Model, Sequential, load_model
from keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from tqdm import tqdm
import zipfile
import random
import math


plot = False
train = False
test = True


def init_model(fs = 10):
    model = Sequential()
    model.add(Dense(800, activation='relu', input_shape=(fs,)))
    model.add(Dense(400, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(400, activation='relu'))
    model.add(Dense(400, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(200, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    return model

def get_event(event):
    zf = zipfile.ZipFile('train_1.zip')
    hits= pd.read_csv(zf.open('train_1/%s-hits.csv'%event))
    cells= pd.read_csv(zf.open('train_1/%s-cells.csv'%event))
    truth= pd.read_csv(zf.open('train_1/%s-truth.csv'%event))
    particles= pd.read_csv(zf.open('train_1/%s-particles.csv'%event))
    return hits, cells, truth, particles


'''
TRAINING SESSION
'''



if train:
    my_randoms = []
    for i in range(80):
        my_randoms.append(random.randrange(100, 1000, 1))

    Train = []
    for i in tqdm(my_randoms):
        event = 'event000001%00d'%i
        hits, cells, truth, particles = get_event(event)
        hit_cells = cells.groupby(['hit_id']).value.count().values
        hit_value = cells.groupby(['hit_id']).value.sum().values
        features = np.hstack((hits[['x','y','z']]/1000, hit_cells.reshape(len(hit_cells),1)/10,hit_value.reshape(len(hit_cells),1)))
        particle_ids = truth.particle_id.unique()
        particle_ids = particle_ids[np.where(particle_ids!=0)[0]]

        pair = []
        for particle_id in particle_ids:
            hit_ids = truth[truth.particle_id == particle_id].hit_id.values-1
            for i in hit_ids:
                for j in hit_ids:
                    if i != j:
                        pair.append([i,j])
        pair = np.array(pair)
        Train1 = np.hstack((features[pair[:,0]], features[pair[:,1]], np.ones((len(pair),1))))

        if len(Train) == 0:
            Train = Train1
        else:
            Train = np.vstack((Train,Train1))

        n = len(hits)
        size = len(Train1)*2
        p_id = truth.particle_id.values
        i =np.random.randint(n, size=size)
        j =np.random.randint(n, size=size)
        pair = np.hstack((i.reshape(size,1),j.reshape(size,1)))
        pair = pair[((p_id[i]==0) | (p_id[i]!=p_id[j]))]

        Train0 = np.hstack((features[pair[:,0]], features[pair[:,1]], np.zeros((len(pair),1))))

        print(event, Train1.shape)

        Train = np.vstack((Train,Train0))
    del Train0, Train1

    np.random.shuffle(Train)
    print(Train.shape)

    model = init_model()
    model.summary()
    checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath='model_80_events_best.hdf5',
                                                      verbose=1, save_best_only=True)
    lr = -7
    model.compile(loss=['binary_crossentropy'], optimizer=Adam(lr=10 ** lr), metrics=['accuracy'])
    History = model.fit(x=Train[:, :-1], y=Train[:, -1], batch_size=32000, epochs=1, verbose=1, callbacks=checkpointer,
                        validation_split=0.15, shuffle=True)

    lr = -5
    model.compile(loss=['binary_crossentropy'], optimizer=Adam(lr=10 ** lr), metrics=['accuracy'])
    history = model.fit(x=Train[:, :-1], y=Train[:, -1], batch_size=32000, epochs=50, verbose=1, callbacks=checkpointer,
                        validation_split=0.15, shuffle=True)

    plt.plot(history.history['loss'], label='Training loss')
    plt.plot(history.history['val_loss'], label='Validation loss')
    plt.title("Loss during the training process")
    plt.xlabel('# epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.show()

    plt.plot(history.history['accuracy'], label='Training accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation accuracy')
    plt.title("Accuracy during the training process")
    plt.xlabel('# epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()
    plt.show()



'''
TESTING SESSION
'''

if test:

    model = load_model('model_80_events.hdf5')

    event = 'event000002001'
    hits, cells, truth, particles = get_event(event)
    hit_cells = cells.groupby(['hit_id']).value.count().values
    hit_value = cells.groupby(['hit_id']).value.sum().values
    features = np.hstack(
        (hits[['x', 'y', 'z']] / 1000, hit_cells.reshape(len(hit_cells), 1) / 10, hit_value.reshape(len(hit_cells), 1)))
    count = hits.groupby(['volume_id', 'layer_id', 'module_id'])['hit_id'].count().values
    module_id = np.zeros(len(hits), dtype='int32')
    print(count)
    print(len(count))

    for i in range(len(count)):
        si = np.sum(count[:i])
        module_id[si:si + count[i]] = i


    def get_path(hit, mask, thr):
        path = [hit]
        a = 0
        while True:
            c = get_predict(path[-1], thr / 2)
            mask = (c > thr) * mask
            mask[path[-1]] = 0

            if 1:
                cand = np.where(c > thr)[0]
                if len(cand) > 0:
                    mask[cand[np.isin(module_id[cand], module_id[path])]] = 0

            a = (c + a) * mask
            if a.max() < thr * len(path):
                break
            path.append(a.argmax())
        return path


    def get_predict(hit, thr=0.5):
        Tx = np.zeros((len(truth), 10))
        Tx[:, 5:] = features
        Tx[:, :5] = np.tile(features[hit], (len(Tx), 1))
        pred = model.predict(Tx, batch_size=len(Tx))[:, 0]
        # TTA
        idx = np.where(pred > thr)[0]
        Tx2 = np.zeros((len(idx), 10))
        Tx2[:, 5:] = Tx[idx, :5]
        Tx2[:, :5] = Tx[idx, 5:]
        pred1 = model.predict(Tx2, batch_size=len(idx))[:, 0]
        pred[idx] = (pred[idx] + pred1) / 2
        return pred


    def closest(lst, K):
        return lst[min(range(len(lst)), key=lambda i: abs(lst[i] - K))]

    def distance_err(path, gt):
        error = 0
        for n_hits in path:
            if n_hits not in gt.tolist():
                re_n_hits = closest(gt.tolist(), n_hits)
                t = truth[truth.hit_id == n_hits + 1]
                re_t = truth[truth.hit_id == re_n_hits + 1]
                dots_dist = (t.tx.item()-re_t.tx.item())**2+(t.ty.item()-re_t.ty.item())**2+(t.tz.item()-re_t.tz.item())**2

                error += math.sqrt(dots_dist)

        return error

    def traj_distance(gt):
        start = sorted(gt.tolist())[0]
        end = sorted(gt.tolist())[-1]
        s = truth[truth.hit_id == start + 1]
        e = truth[truth.hit_id == end + 1]
        trajectory_length = math.sqrt((s.tx.item()-e.tx.item())**2+(s.ty.item()-e.ty.item())**2+(s.tz.item()-e.tz.item())**2)
        return trajectory_length


    def percentage_err(err, length):
        if length == 0:
            return 0
        else:
            perc_err =err/length
            return perc_err*100

    all_errs = []
    skipped = 0
    for hit in range(51,52):
        path = get_path(hit, np.ones(len(truth)), 0.95)
        gt = np.where(truth.particle_id == truth.particle_id[hit])[0]

        if len(gt.tolist()) > 15:
            skipped +=1
            continue
        ordered = sorted(path)

        if ordered[0] == 0:
            skipped+=1
            continue
        last_gt = sorted(gt.tolist())[-1]

        for el in path:
            if el > last_gt+50:
                path[path.index(el)] = 0

        path = list(filter(lambda num: num != 0, path))

        de = distance_err(path,gt)
        td = traj_distance(gt)
        print('hit_id = ', hit + 1)
        print('reconstruct :', sorted(path))
        print('ground truth:', sorted(gt.tolist()))
        print('distance error: ', de)
        print('trajectory length: ', td)
        print('percentage error: ', percentage_err(de,td),  '%')
        print('               ')
        if percentage_err(de,td) <100:
            all_errs.append(percentage_err(de,td))
    print(all_errs)
    print(sum(all_errs))
    print(len(all_errs))
    print(skipped)
    avg_err = sum(all_errs)/(len(all_errs))
    print('avg percentage error is ', avg_err, '%')



    '''PLOT'''
    plt.figure(figsize=(15, 15))
    ax = plt.axes(projection='3d')
    tx_re = []
    ty_re = []
    tz_re = []

    for n_hits in sorted(path):
        t = truth[truth.hit_id == n_hits+1]
        tx_re.append(float(t.tx))
        ty_re.append(float(t.ty))
        tz_re.append(float(t.tz))

    tx_tr = []
    ty_tr = []
    tz_tr = []

    for n_hits in gt.tolist():
        t = truth[truth.hit_id == n_hits +1]
        tx_tr.append(float(t.tx))
        ty_tr.append(float(t.ty))
        tz_tr.append(float(t.tz))

    ax.plot3D(tz_re, tx_re, ty_re)
    ax.plot3D(tz_tr, tx_tr, ty_tr)
    ax.set_xlabel('z (mm)')
    ax.set_ylabel('x (mm)')
    ax.set_zlabel('y (mm)')
    # These two added to widen the 3D space
    #ax.scatter(3000, 3000, 3000, s=0)
    #ax.scatter(-3000, -3000, -3000, s=0)
    plt.show()


'''
PLOTTING SESSION
'''


# Get every 100th particle
if plot:
    event = 'event000001001'
    hits, cells, truth, particles = get_event(event)
    tracks = truth.particle_id.unique()[1::100]

    plt.figure(figsize=(15, 15))
    ax = plt.axes(projection='3d')
    for track in tracks:
        t = truth[truth.particle_id == track]
        ax.plot3D(t.tz, t.tx, t.ty)
    ax.set_xlabel('z (mm)')
    ax.set_ylabel('x (mm)')
    ax.set_zlabel('y (mm)')
    # These two added to widen the 3D space
    ax.scatter(3000, 3000, 3000, s=0)
    ax.scatter(-3000, -3000, -3000, s=0)
    plt.show()
