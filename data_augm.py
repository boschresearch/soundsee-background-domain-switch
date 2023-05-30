""" Utility classes and functions related to Background-Domain-Switch (Interspeech 2023).
Copyright (c) 2023 Robert Bosch GmbH
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.
You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
import torch
import numpy as np


def pseudo_strong_labeling(data, model, th, iter_num, norm, scale):
    """Pseudo strong labeling for weak-labeled or unlabeled data based on the trained SED model.
    The model is expected to have strong label predictions as the first output, 
    e.g., strong, weak, ... = model(data)
    input shape of data= [batch, dim, time]

    Returns:
        numpy.Array of binary strong labels, shape=[batch, class, time]
    """
    # feature pre-processing functions if any
    if scale:
        data = scale(data) # e.g., log scale
    if norm:
        data = norm(data) # e.g., z-norm
    pseudo_strong = []
    for _ in range(iter_num):
        pseudo_strong.append(model(data)[0]) # gets the strong predictions
    pseudo_strong = torch.mean(torch.stack(pseudo_strong), dim=0)
    pseudo_strong = pseudo_strong.cpu().detach().numpy()
    # binarize labels based on the defined threshold
    pseudo_strong[pseudo_strong>=th] = 1
    pseudo_strong[pseudo_strong<th] = 0
    return pseudo_strong.astype('int')


def parse_noevent_segment(strong_label, min_frames, upsample):
    """Parsing background segments that contain zero-target events based on strong labels.
    input shape of strong label= [batch, class, time]
    
    Returns:
        list of segment starting and ending indexes
    """
    event_detected = np.sum(strong_label, axis=1)
    NoEvent_SegIdx = []
    for i in range(len(event_detected)):
        zero_event_idxs = np.where(event_detected[i]==0)[0]
        if len(zero_event_idxs)!=0:
            # find segmentation points
            _diff = np.where(np.diff(zero_event_idxs)!=1)[0] + 1
            if 0 not in _diff:
                _diff = np.insert(_diff, 0, 0)
            # output zero-event slices
            zero_event_segment = []
            for j in range(len(_diff)):
                try:
                    seg_start_idx = _diff[j]
                    seg_end_idx = _diff[j+1]
                    zero_event_segment.append(zero_event_idxs[seg_start_idx:seg_end_idx])
                except:
                    zero_event_segment.append(zero_event_idxs[_diff[j]:])
            zero_event_segment_Idx = []
            for j in range(len(zero_event_segment)):
                # upsampling back to the original length
                seg_start_idx = zero_event_segment[j][0]*upsample
                seg_end_idx = seg_start_idx + len(zero_event_segment[j])*upsample
                # checking if the segment is long enough
                if (seg_end_idx-seg_start_idx) > min_frames:
                    zero_event_segment_Idx.append([seg_start_idx, seg_end_idx])
            NoEvent_SegIdx.append(zero_event_segment_Idx)
        else:
            NoEvent_SegIdx.append([])
    return NoEvent_SegIdx


def switch_background(data, domain_seg, background_seg_pool):
    """Switch the background segments of source domain data (i.e., data & domain_seg) to randomly 
    selected backgrounds from another domain's background pool.
    input shape of data= [batch, dim, time]

    Returns:
        torch.Tensor, same shape as the input data after in-place switch operation
    """
    for i in range(len(data)):
        if len(domain_seg[i])!=0:
            for j in range(len(domain_seg[i])):
                _seg_start_idx = domain_seg[i][j][0]
                _seg_end_idx = domain_seg[i][j][1]

                rdn_pool_idx = np.random.choice(len(background_seg_pool), size=1, replace=True)
                target_back = background_seg_pool[int(rdn_pool_idx)]

                # shorter background case: repeat & crop
                if data[i,:,_seg_start_idx:_seg_end_idx].size()[-1] > target_back.size()[-1]:
                    repeat_times = int(np.ceil(data[i,:,_seg_start_idx:_seg_end_idx].size()[-1]/target_back.size()[-1]))
                    target_back = target_back.repeat(1, repeat_times) # repeat & crop
                    target_back = target_back[:, 0:_seg_end_idx-_seg_start_idx]

                # longer background case: directly crop
                elif data[i,:,_seg_start_idx:_seg_end_idx].size()[-1] < target_back.size()[-1]:
                    target_back = target_back[:, 0:_seg_end_idx-_seg_start_idx]

                # switch background
                data[i,:,_seg_start_idx:_seg_end_idx] = target_back
    return data


def BDS(feats,
        labels_strong,
        set_masks,
        model,
        norm=None,
        scale=None,
        seq_pooling_factor=4,
        event_threshold=0.4,
        min_frames=40,
        bidirectional=False,
        stochastic_iter=1):
    """Background-Domain-Switch (BDS) data augmentation for SED is performing to switch zero-target-event 
        background segments between different domains (e.g., synth or real-world) based on pseudo-labeling.
    
    Args:
        feats: input batch features tensor, shape=[batch, dim, time], e.g., mel-spec
        labels_strong: tensor of strong labels for the synth train set, shape=[batch, class, time]
        set_masks: list of tensor masks, [strong_mask, weak_mask, unlabeled_mask]
        model: trained torch SED model from the previous training iteration, e.g., CRNN
        norm: input features normalization if any
        scale: apply scale on input features if any
        seq_pooling_factor: int, sequence temporal pooling factor of the model
        event_threshold (𝜏): float, decision threshold to trigger detected event (0 ~ 1)
        min_frames (m): int, minimum window size for the background segment candidates
        bidirectional: bool, if True, do domain-switch bidirectionally
        stochastic_iter: int, number of stochastic predictions for pseudo-labeling

    Returns:
        torch.Tensor of input feats after switched background domains (in-place operation)
    """
    with torch.no_grad():
        strong_mask, weak_mask, unlabeled_mask = set_masks

        # init data & labels based on different sets in a batch
        data_synth, data_weak, data_unlabel = feats[strong_mask], feats[weak_mask], feats[unlabeled_mask]
        synth_strong = labels_strong[strong_mask].cpu().detach().numpy().astype('int')

        # stochastic pseudo-strong-labeling on the weak and unlabeled sets
        weak_pseudo_strong = pseudo_strong_labeling(data_weak, model, th=event_threshold, iter_num=stochastic_iter, norm=norm, scale=scale)
        unlabel_pseudo_strong = pseudo_strong_labeling(data_unlabel, model, th=event_threshold, iter_num=stochastic_iter, norm=norm, scale=scale)

        # obtain zero-event background segment index
        domain_seg_synth = parse_noevent_segment(synth_strong, min_frames=min_frames, upsample=seq_pooling_factor)
        domain_seg_weak = parse_noevent_segment(weak_pseudo_strong, min_frames=min_frames, upsample=seq_pooling_factor)
        domain_seg_unlabel = parse_noevent_segment(unlabel_pseudo_strong, min_frames=min_frames, upsample=seq_pooling_factor)

        # generate zero-event background segment pool
        background_seg_pool = []
        for i in range(len(data_weak)): # background segments from weak set
            if len(domain_seg_weak[i])!=0:
                for j in range(len(domain_seg_weak[i])):
                    _seg_start_idx = domain_seg_weak[i][j][0]
                    _seg_end_idx = domain_seg_weak[i][j][1]
                    background_seg_pool.append(data_weak[i,:,_seg_start_idx:_seg_end_idx])

        for i in range(len(data_unlabel)): # background segments from unlabeled set
            if len(domain_seg_unlabel[i])!=0:
                for j in range(len(domain_seg_unlabel[i])):
                    _seg_start_idx = domain_seg_unlabel[i][j][0]
                    _seg_end_idx = domain_seg_unlabel[i][j][1]
                    background_seg_pool.append(data_unlabel[i,:,_seg_start_idx:_seg_end_idx])
        
        # synth data switch to real-world background
        if len(background_seg_pool)!=0: # make sure we have something in the background pool to select
            data_synth_switched = switch_background(data_synth, domain_seg_synth, background_seg_pool)
        else:
            data_synth_switched = feats[strong_mask]
        
        # do domain-switch to real-world data as well 
        # (i.e., switch real-world background to synth background)
        if bidirectional:
            # zero-event background pool based on the synth set
            background_seg_pool = [] # reset background pool
            for i in range(len(data_synth)): # background segments from synth set
                if len(domain_seg_synth[i])!=0:
                    for j in range(len(domain_seg_synth[i])):
                        _seg_start_idx = domain_seg_synth[i][j][0]
                        _seg_end_idx = domain_seg_synth[i][j][1]
                        background_seg_pool.append(data_synth[i,:,_seg_start_idx:_seg_end_idx])
            
            # real-world data switch to synth background
            if len(background_seg_pool)!=0: # make sure we have something in the background pool to select
                data_weak_switched = switch_background(data_weak, domain_seg_weak, background_seg_pool)
                data_unlabel_switched = switch_background(data_unlabel, domain_seg_unlabel, background_seg_pool)
            else:
                data_weak_switched = feats[weak_mask]
                data_unlabel_switched = feats[unlabeled_mask]
            
            feats[weak_mask] = data_weak_switched
            feats[unlabeled_mask] = data_unlabel_switched
            
        feats[strong_mask] = data_synth_switched
        return feats

