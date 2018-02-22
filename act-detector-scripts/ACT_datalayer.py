# -*- coding: utf-8 -*-
import sys
import os
import random

import numpy as np
import cv2

from ACT_utils import iou2d
from Dataset import GetDataset
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))
import caffe

#进行photometric distortions的参数
distort_params = {
    'brightness_prob': 0.5,
    'brightness_delta': 32,
    'contrast_prob': 0.5,
    'contrast_lower': 0.5,
    'contrast_upper': 1.5,
    'hue_prob': 0.5,
    'hue_delta': 18,
    'saturation_prob': 0.5,
    'saturation_lower': 0.5,
    'saturation_upper': 1.5,
    'random_order_prob': 0.0,
}

expand_params = {
    'expand_prob': 0.5,
    'max_expand_ratio': 4.0,
}

#采样用的参数(参考SSD的data augmentation)
batch_samplers = [{
    'sampler': {},
    'max_trials': 1,
    'max_sample': 1,
}, {
    'sampler': {'min_scale': 0.3, 'max_scale': 1.0, 'min_aspect_ratio': 0.5,  'max_aspect_ratio': 2.0,},
    'sample_constraint': {'min_jaccard_overlap': 0.1, },
    'max_trials': 50,
    'max_sample': 1,
}, {
    'sampler': {'min_scale': 0.3, 'max_scale': 1.0, 'min_aspect_ratio': 0.5, 'max_aspect_ratio': 2.0,},
    'sample_constraint': {'min_jaccard_overlap': 0.3,},
    'max_trials': 50,
    'max_sample': 1,
},{
    'sampler': {'min_scale': 0.3, 'max_scale': 1.0, 'min_aspect_ratio': 0.5, 'max_aspect_ratio': 2.0,},
    'sample_constraint': {'min_jaccard_overlap': 0.5,},
    'max_trials': 50,
    'max_sample': 1,
},{
    'sampler': {'min_scale': 0.3, 'max_scale': 1.0, 'min_aspect_ratio': 0.5, 'max_aspect_ratio': 2.0,},
    'sample_constraint': {'min_jaccard_overlap': 0.7,},
    'max_trials': 50,
    'max_sample': 1,
},{
    'sampler': {'min_scale': 0.3, 'max_scale': 1.0, 'min_aspect_ratio': 0.5, 'max_aspect_ratio': 2.0,},
    'sample_constraint': {'min_jaccard_overlap': 0.9,},
    'max_trials': 50,
    'max_sample': 1,
},{
    'sampler': {'min_scale': 0.3, 'max_scale': 1.0, 'min_aspect_ratio': 0.5, 'max_aspect_ratio': 2.0,},
    'sample_constraint': {'max_jaccard_overlap': 1.0,},
    'max_trials': 50,
    'max_sample': 1,
},]

#随机修改亮度
def random_brightness(imglist, brightness_prob, brightness_delta):
    if random.random() < brightness_prob:
        brig = random.uniform(-brightness_delta, brightness_delta)
        for i in xrange(len(imglist)):
            imglist[i] += brig

    return imglist

#随机修改对比度
def random_contrast(imglist, contrast_prob, contrast_lower, contrast_upper):
    if random.random() < contrast_prob:
        cont = random.uniform(contrast_lower, contrast_upper)
        for i in xrange(len(imglist)):
            imglist[i] *= cont
    
    return imglist

#随机修改饱和度
def random_saturation(imglist, saturation_prob, saturation_lower, saturation_upper):
    if random.random() < saturation_prob:
        satu = random.uniform(saturation_lower, saturation_upper)
        for i in xrange(len(imglist)):
            hsv = cv2.cvtColor(imglist[i], cv2.COLOR_BGR2HSV)
            hsv[:, :, 1] *= satu
            imglist[i] = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return imglist

#随机修改色调
def random_hue(imglist, hue_prob, hue_delta):
    if random.random() < hue_prob:
        hue = random.uniform(-hue_delta, hue_delta)
        for i in xrange(len(imglist)):
            hsv = cv2.cvtColor(imglist[i], cv2.COLOR_BGR2HSV)
            hsv[:, :, 0] += hue
            imglist[i] = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return imglist

#对imglist进行photometric distortions
def apply_distort(imglist, distort_param):
    out_imglist = imglist

    if distort_param['random_order_prob'] != 0: raise NotImplementedError

    if random.random() > 0.5:
        out_imglist = random_brightness(out_imglist, distort_param['brightness_prob'], distort_param['brightness_delta'])
        out_imglist = random_contrast(out_imglist, distort_param['contrast_prob'], distort_param['contrast_lower'], distort_param['contrast_upper'])
        out_imglist = random_saturation(out_imglist, distort_param['saturation_prob'], distort_param['saturation_lower'], distort_param['saturation_upper'])
        out_imglist = random_hue(out_imglist, distort_param['hue_prob'], distort_param['hue_delta'])
    else:
        out_imglist = random_brightness(out_imglist, distort_param['brightness_prob'], distort_param['brightness_delta'])
        out_imglist = random_saturation(out_imglist, distort_param['saturation_prob'], distort_param['saturation_lower'], distort_param['saturation_upper'])
        out_imglist = random_hue(out_imglist, distort_param['hue_prob'], distort_param['hue_delta'])
        out_imglist = random_contrast(out_imglist, distort_param['contrast_prob'], distort_param['contrast_lower'], distort_param['contrast_upper'])

    return out_imglist

#实施对小物体的数据增广(参考SSD)
def apply_expand(imglist, tubes, expand_param, mean_values=None):
    # Tubes: dict of label -> list of tubes with tubes being <x1> <y1> <x2> <y2>
    out_imglist = imglist
    out_tubes = tubes

    if random.random() < expand_param['expand_prob']:
        #对小物体的数据增广方法(参考SSD)
        expand_ratio = random.uniform(1, expand_param['max_expand_ratio'])
        oh,ow = imglist[0].shape[:2]
        h = int(oh * expand_ratio)
        w = int(ow * expand_ratio)
        out_imglist = [np.zeros((h, w, 3), dtype=np.float32) for i in xrange(len(imglist))]
        h_off = int(np.floor(h - oh))
        w_off = int(np.floor(w - ow))
        if mean_values is not None:
            for i in xrange(len(imglist)):
                out_imglist[i] += np.array(mean_values).reshape(1, 1, 3)
        for i in xrange(len(imglist)):
            out_imglist[i][h_off:h_off+oh, w_off:w_off+ow, :] = imglist[i]
    
        # project boxes
        #增广之后，相应的tube的位置要进行修改#
        for ilabel in tubes:
            for itube in xrange(len(tubes[ilabel])):
                out_tubes[ilabel][itube] += np.array([[w_off, h_off, w_off, h_off]], dtype=np.float32)

    return out_imglist, out_tubes


#--------------------对图像进行采样--------------------#
#选择要采样的位置(类似SSD中对patch的选择)
def sample_cuboids(tubes, batch_samplers, imheight, imwidth):
    sampled_cuboids = []
    for batch_sampler in batch_samplers:
        max_trials = batch_sampler['max_trials']
        max_sample = batch_sampler['max_sample']
        itrial = 0 #表示对某种batch_sampler的实验次数
        isample = 0 #控制是否对某种batch_sampler继续取样
        sampler = batch_sampler['sampler']

        min_scale = sampler['min_scale'] if 'min_scale' in sampler else 1
        max_scale = sampler['max_scale'] if 'max_scale' in sampler else 1
        min_aspect = sampler['min_aspect_ratio'] if 'min_aspect_ratio' in sampler else 1
        max_aspect = sampler['max_aspect_ratio'] if 'max_aspect_ratio' in sampler else 1

        while itrial < max_trials and isample < max_sample:
            # sample a normalized box
            scale = random.uniform(min_scale, max_scale)
            aspect = random.uniform(min_aspect, max_aspect)
            width = scale * np.sqrt(aspect)
            height = scale / np.sqrt(aspect)
            x = random.uniform(0, 1 - width)
            y = random.uniform(0, 1 - height)

            # rescale the box
            sampled_cuboid = np.array([x*imwidth, y*imheight, (x+width)*imwidth, (y+height)*imheight], dtype=np.float32)
            
            # check constraint
            itrial += 1
            if not 'sample_constraint' in batch_sampler:
                sampled_cuboids.append(sampled_cuboid)
                isample += 1
                continue

            constraints = batch_sampler['sample_constraint']
            #计算gt的每一个tube与sampled_cuboid的平均重叠度，ious为平均重叠度数组
            ious = np.array([np.mean(iou2d(t, sampled_cuboid)) for t in sum(tubes.values(),[])])
            
            #empty gt
            if ious.size == 0: 
                isample += 1
                continue
            
            #判断所采样的cuboid是否满足重叠度相关的条件
            #cuboid只要与某一个gt tube的重叠度不小于min_jaccard_overlap，就保存下来
            if 'min_jaccard_overlap' in constraints and ious.max() >= constraints['min_jaccard_overlap']:
                sampled_cuboids.append( sampled_cuboid )
                isample += 1
                continue

            if 'max_jaccard_overlap' in constraints and ious.min() >= constraints['max_jaccard_overlap']:
                sampled_cuboids.append( sampled_cuboid )
                isample += 1
                continue

    return sampled_cuboids

#实施采样
def crop_image(imglist, tubes, batch_samplers):
    candidate_cuboids = sample_cuboids(tubes, batch_samplers, imglist[0].shape[0], imglist[0].shape[1])

    if not candidate_cuboids:
        return imglist, tubes
    
    #random.choice表示随机选取内容
    crop_cuboid = random.choice(candidate_cuboids)
    x1, y1, x2, y2 = map(int, crop_cuboid.tolist())
    
    #裁剪cuboid对应在img上的位置的部分存入imglist
    for i in xrange(len(imglist)):
        imglist[i] = imglist[i][y1:y2+1, x1:x2+1, :]

    #keep the overlapped part of the groudtruth tube if the center of it is in the sampled cuboid(参考SSD)
    out_tubes = {}
    wi = x2 - x1
    hi = y2 - y1
    
    for ilabel in tubes:
        for itube in xrange(len(tubes[ilabel])):
            t = tubes[ilabel][itube]
            t -= np.array([[x1, y1, x1, y1]], dtype=np.float32)

            # check if valid
            cx = 0.5 * (t[:, 0] + t[:, 2])
            cy = 0.5 * (t[:, 1] + t[:, 3])

            if np.any(cx < 0) or np.any(cy < 0) or np.any(cx > wi) or np.any(cy > hi):
                continue

            if not ilabel in out_tubes:
                out_tubes[ilabel] = []

            # clip box
            #取重叠部分
            t[:, 0] = np.maximum(0, t[:, 0])
            t[:, 1] = np.maximum(0, t[:, 1])
            t[:, 2] = np.minimum(wi, t[:, 2])
            t[:, 3] = np.minimum(hi, t[:, 3])

            out_tubes[ilabel].append(t)

    return imglist, out_tubes
#------------------------------------------------------#

# Assisting function for finding a good/bad tubelet
def tubelet_in_tube(tube, i, K):
    # True if all frames from i to (i + K - 1) are inside tube
    # it's sufficient to just check the first and last frame. 
    return (i in tube[: ,0] and i + K - 1 in tube[:, 0])

def tubelet_out_tube(tube, i, K): 
    # True if all frames between i and (i + K - 1) are outside of tube
    return all([not j in tube[:, 0] for j in xrange(i, i + K)])

def tubelet_in_out_tubes(tube_list, i, K): 
    # Given a list of tubes: tube_list, return True if  
    # all frames from i to (i + K - 1) are either inside (tubelet_in_tube)
    # or outside (tubelet_out_tube) the tubes. 
    return all([tubelet_in_tube(tube, i, K) or tubelet_out_tube(tube, i, K) for tube in tube_list])

def tubelet_has_gt(tube_list, i, K):
    # Given a list of tubes: tube_list, return True if  
    # the tubelet starting spanning from [i to (i + K - 1)]
    # is inside (tubelet_in_tube) at least a tube in tube_list. 
    return any([tubelet_in_tube(tube, i, K) for tube in tube_list])

#--------------------自定义的数据层--------------------#
#该层没有权值，反向传播不需要进行权值的更新。如果需要定义更新自身权值的层，最好使用C++
class MultiframesLayer(caffe.Layer):

    def shuffle(self): # shuffle the list of possible starting frames
        self._order = range(self._nseqs)
        if self._shuffle:
            # set seed like that to have exactly the same shuffle even if we restart from a caffemodel
            random.seed(self._rand_seed + self._nshuffles)
            random.shuffle(self._order)
        self._nshuffles += 1
        self._next = 0

    #在网络运行之前根据相关参数对layer进行初始化
    def setup(self, bottom, top):
        layer_params = eval(self.param_str)

        assert 'dataset_name' in layer_params
        dataset_name = layer_params['dataset_name']
        self._dataset = GetDataset(dataset_name)

        assert 'K' in layer_params
        self._K = layer_params['K']
        assert self._K > 0

        # parse optional argument
        default_values = {
            'rand_seed': 0,
            'shuffle': True,
            'batch_size': 32 // self._K,
            'mean_values': [104, 117, 123],
            'resize_height': 300,
            'resize_width': 300,
            'restart_iter': 0,
            'flow': False,
            'ninput': 1,
        }

        for k in default_values.keys():
            if k in layer_params:
                lay_param = layer_params[k]
            else:
                lay_param = default_values[k]
            setattr(self, '_' + k, lay_param)

        if not self._flow and self._ninput > 1:
            raise NotImplementedError("ACT-detector: Not implemented: ninput > 1 with rgb frames")

        d = self._dataset
        K = self._K

        # build index (v,i) of valid starting chunk
        self._indices = []
        for v in d.train_vlist():
            vtubes = sum(d.gttubes(v).values(), [])
            #for training, we consider only sequences of frames in which all frames contain the ground-truth action 
            self._indices += [(v,i) for i in range(1, d.nframes(v)+2-K) if tubelet_in_out_tubes(vtubes,i,K) and tubelet_has_gt(vtubes,i,K)]
            
        self._nseqs = len(self._indices)

        self._iter = 0
        self._nshuffles = 0
        self.shuffle()

        if self._restart_iter > 0:
            assert self._next == 0

            self._iter = self._restart_iter
            iimages = self._restart_iter * self._batch_size

            while iimages > self._nseqs:
                self.shuffle()
                iimages -= self._nseqs

            self._next = iimages

        for i in xrange(K):
            top[i].reshape(self._batch_size, 3 * self._ninput, self._resize_height, self._resize_width)

        #表示label
        top[K].reshape(1, 1, 1, 8)

    def prepare_blob(self):
        d = self._dataset
        K = self._K

        # Have the same data augmentation, even if restarted
        random.seed(self._rand_seed + self._iter)

        data = [np.empty((self._batch_size, 3 * self._ninput, self._resize_height, self._resize_width), dtype=np.float32) for ii in range(K)]

        alltubes = []
        for i in xrange(self._batch_size):
            if self._next == self._nseqs:
                self.shuffle()

            v,frame = self._indices[self._order[self._next]]
            # flipping with probability 0.5
            do_mirror = random.getrandbits(1) == 1

            # load images and tubes and apply mirror
            images = []
            if self._flow:
                images = [cv2.imread(d.flowfile(v, min(frame+ii, d.nframes(v)))).astype(np.float32) for ii in range(K + self._ninput - 1)]
            else:
                images = [cv2.imread(d.imfile(v, frame+ii)).astype(np.float32) for ii in range(K)]
            
            #image的水平翻转
            if do_mirror:
                #::-1表示将width上的元素倒序排，即水平翻转
                images = [im[:, ::-1, :] for im in images]
                # reverse the x component of the flow
                if self._flow:
                    for ii in xrange(K + self._ninput - 1):
                        images[ii][:, :, 2] = 255 - images[ii][:, :, 2]

            h, w = d.resolution(v)
            #截取与images对应长度的满足条件的gt tubes存入TT
            TT = {}
            for ilabel, tubes in d.gttubes(v).iteritems():
                for t in tubes:
                    if frame not in t[:, 0]:
                        continue

                    assert frame + K - 1 in t[:, 0]
                    #对应tube的水平翻转
                    if do_mirror:
                        # copy otherwise it will change the gt of the dataset also
                        t = t.copy()
                        xmin = w - t[:, 3]
                        t[:, 3] = w - t[:, 1]
                        t[:, 1] = xmin

                    boxes = t[(t[:, 0] >= frame) * (t[:, 0] < frame + K) ,1:5]
                    assert boxes.shape[0] == K

                    if ilabel not in TT:
                        TT[ilabel] = []

                    TT[ilabel].append( boxes)

            # apply data augmentation
            images = apply_distort(images, distort_params)
            images, TT = apply_expand(images, TT, expand_params, mean_values=self._mean_values)
            images, TT = crop_image(images, TT, batch_samplers)
            hi,wi = images[0].shape[:2]

            # resize
            images = [cv2.resize(im, (self._resize_width, self._resize_height), interpolation=cv2.INTER_LINEAR) for im in images]
            for ii in range(K):
                for iii in xrange(self._ninput):
                    data[ii][i, 3*iii:3*iii + 3, :, :] = np.transpose( images[ii + iii], (2, 0, 1))

            idxtube = 0
            for ilabel in TT:
                for itube in xrange(len(TT[ilabel])):
                    for b in TT[ilabel][itube]:
                        alltubes.append([i, ilabel+1, idxtube, b[0]/wi, b[1]/hi, b[2]/wi, b[3]/hi, 0])
                    idxtube += 1
            self._next += 1
        self._iter += 1
        
        #数据去均值化操作
        for ii in range(K):
            #tile()函数表示将array在对应方向上重复若干辞
            data[ii] -= np.tile(np.array(self._mean_values, dtype=np.float32)[None, :, None, None], (1, self._ninput, 1, 1))

        label = np.array(alltubes, dtype=np.float32)

        # label shape 1x1x1x8; if no boxes, then -1
        if label.size == 0:
            label = -np.ones((1, 1, 1, 8), dtype=np.float32)
        else:
            label = label.reshape(1, 1, -1, 8)#-1表示自动计算该维的大小

        return data + [label]

    #网络的前向传播
    def forward(self, bottom, top):
        blobs = self.prepare_blob()
        for ii in xrange(len(top) - 1):
            top[ii].data[...] = blobs[ii].astype(np.float32, copy=False)
        # *表示将元组解包
        top[len(top) - 1].reshape(*(blobs[len(top) - 1].shape))
        top[len(top) - 1].data[...] = blobs[len(top) - 1].astype(np.float32, copy=False)

    #网络的后向传播
    def backward(self, bottom, propagate_down, top):
        pass

    #在forward之前之前调用，根据bottom blob的尺寸调整中间变量和top blob的尺寸
    def reshape(self, bottom, top):
        # done in the forward
        pass
#------------------------------------------------------#
