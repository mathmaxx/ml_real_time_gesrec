import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable

from utils import AverageMeter


def calculate_video_results(output_buffer, video_id, test_results, class_names):
    video_outputs = torch.stack(output_buffer)
    average_scores = torch.mean(video_outputs, dim=0)
    sorted_scores, locs = torch.topk(average_scores, k=10)

    video_results = []
    for i in range(sorted_scores.size(0)):
        video_results.append({
            'label': class_names[int(locs[i])],
            'score': float(sorted_scores[i])
        })

    test_results['results'][video_id] = video_results


def test(data_loader, model, opt, class_names):
    print('test')

    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()

    end_time = time.time()
    output_buffer = []
    previous_video_id = ''

    if opt.save_result == 'json':
        test_results = {'results': {}}
    else:
        test_results = {'results': {'sum': {}, 'mean': {}}}
    for i, (inputs, targets) in enumerate(data_loader):
        data_time.update(time.time() - end_time)

        with torch.no_grad():

            inputs = Variable(inputs)
            outputs = model(inputs)
        if not opt.no_softmax_in_test:
            outputs = F.softmax(outputs, dim=1)

        for j in range(outputs.size(0)):
            if not (i == 0 and j == 0) and targets[j] != previous_video_id:
                if opt.save_result == 'json':
                    calculate_video_results(output_buffer, previous_video_id,
                                            test_results, class_names)
                else:
                    test_results['results']['sum'][previous_video_id] = np.array(output_buffer).sum(0)
                    test_results['results']['mean'][previous_video_id] = np.array(output_buffer).mean(0)
                output_buffer = []

            if opt.save_result == 'json':
                output_buffer.append(outputs[j].cpu().numpy().reshape(-1, ))
                previous_video_id = targets[j].item()
            else:
                output_buffer.append(outputs[j].data.cpu())

                previous_video_id = targets[j]

        # if (i % 100) == 0:
        # with open(os.path.join(opt.result_path, '{}.dump'.format(
        #             opt.test_subset)), 'wb') as f:
        #     pickle.dump(test_results, f)

        # with open(
        #         os.path.join(opt.result_path, '{}.json'.format(
        #             opt.test_subset)), 'w') as f:
        #     json.dump(test_results, f)

        batch_time.update(time.time() - end_time)
        end_time = time.time()

        print('[{}/{}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'.format(
            i + 1,
            len(data_loader),
            batch_time=batch_time,
            data_time=data_time))
    with open(
            os.path.join(opt.result_path, '{}.json'.format(opt.test_subset)),
            'w') as f:
        json.dump(test_results, f)
