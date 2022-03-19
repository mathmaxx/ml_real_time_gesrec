import os
import glob
import json
import pandas as pd
import numpy as np
import csv
import torch
import time
from torch.autograd import Variable
from PIL import Image
import cv2
from torch.nn import functional as F

from opts import parse_opts_online
from model import generate_model
from mean import get_mean, get_std
from spatial_transforms import *
from temporal_transforms import *
from target_transforms import ClassLabel
from dataset import get_online_data
from utils import  AverageMeter, LevenshteinDistance, Queue

import pdb
import datetime

import argparse
import asyncio
import logging
import ssl
import uuid
from aiohttp import web
import av
import queue
from collections import deque
import threading
import itertools
from typing import Generic, List, Optional, Union
import traceback
import sys

from aiohttp import web
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaBlackhole, MediaPlayer, MediaRecorder, MediaRelay

###Pretrained RGB models_loc
##Google Drive
#https://drive.google.com/file/d/1V23zvjAKZr7FUOBLpgPZkpHGv8_D-cOs/view?usp=sharing
##Baidu Netdisk
#https://pan.baidu.com/s/114WKw0lxLfWMZA6SYSSJlw code:p1va



def weighting_func(x):
    return (1 / (1 + np.exp(-0.2 * (x - 9))))


def load_models(opt):
    opt.resume_path = opt.resume_path_det
    opt.pretrain_path = opt.pretrain_path_det
    opt.sample_duration = opt.sample_duration_det
    opt.model = opt.model_det
    opt.model_depth = opt.model_depth_det
    opt.width_mult = opt.width_mult_det
    opt.modality = opt.modality_det
    opt.resnet_shortcut = opt.resnet_shortcut_det
    opt.n_classes = opt.n_classes_det
    opt.n_finetune_classes = opt.n_finetune_classes_det

    if opt.root_path != '':
        opt.video_path = os.path.join(opt.root_path, opt.video_path)
        opt.annotation_path = os.path.join(opt.root_path, opt.annotation_path)
        opt.result_path = os.path.join(opt.root_path, opt.result_path)
        if opt.resume_path:
            opt.resume_path = os.path.join(opt.root_path, opt.resume_path)
        if opt.pretrain_path:
            opt.pretrain_path = os.path.join(opt.root_path, opt.pretrain_path)

    opt.scales = [opt.initial_scale]
    for i in range(1, opt.n_scales):
        opt.scales.append(opt.scales[-1] * opt.scale_step)
    opt.arch = '{}'.format(opt.model)
    opt.mean = get_mean(opt.norm_value)
    opt.std = get_std(opt.norm_value)

    print(opt)
    with open(os.path.join(opt.result_path, 'opts_det.json'), 'w') as opt_file:
        json.dump(vars(opt), opt_file)

    torch.manual_seed(opt.manual_seed)

    detector, parameters = generate_model(opt)
    detector = detector.cuda()
    if opt.resume_path:
        opt.resume_path = os.path.join(opt.root_path, opt.resume_path)
        print('loading checkpoint {}'.format(opt.resume_path))
        checkpoint = torch.load(opt.resume_path)

        detector.load_state_dict(checkpoint['state_dict'])

    print('Model 1 \n', detector)
    pytorch_total_params = sum(p.numel() for p in detector.parameters() if
                               p.requires_grad)
    print("Total number of trainable parameters: ", pytorch_total_params)

    opt.resume_path = opt.resume_path_clf
    opt.pretrain_path = opt.pretrain_path_clf
    opt.sample_duration = opt.sample_duration_clf
    opt.model = opt.model_clf
    opt.model_depth = opt.model_depth_clf
    opt.width_mult = opt.width_mult_clf
    opt.modality = opt.modality_clf
    opt.resnet_shortcut = opt.resnet_shortcut_clf
    opt.n_classes = opt.n_classes_clf
    opt.n_finetune_classes = opt.n_finetune_classes_clf
    if opt.root_path != '':
        opt.video_path = os.path.join(opt.root_path, opt.video_path)
        opt.annotation_path = os.path.join(opt.root_path, opt.annotation_path)
        opt.result_path = os.path.join(opt.root_path, opt.result_path)
        if opt.resume_path:
            opt.resume_path = os.path.join(opt.root_path, opt.resume_path)
        if opt.pretrain_path:
            opt.pretrain_path = os.path.join(opt.root_path, opt.pretrain_path)

    opt.scales = [opt.initial_scale]
    for i in range(1, opt.n_scales):
        opt.scales.append(opt.scales[-1] * opt.scale_step)
    opt.arch = '{}'.format(opt.model)
    opt.mean = get_mean(opt.norm_value)
    opt.std = get_std(opt.norm_value)

    print(opt)
    with open(os.path.join(opt.result_path, 'opts_clf.json'), 'w') as opt_file:
        json.dump(vars(opt), opt_file)

    torch.manual_seed(opt.manual_seed)
    classifier, parameters = generate_model(opt)
    classifier = classifier.cuda()
    if opt.resume_path:
        print('loading checkpoint {}'.format(opt.resume_path))
        checkpoint = torch.load(opt.resume_path)

        classifier.load_state_dict(checkpoint['state_dict'])

    print('Model 2 \n', classifier)
    pytorch_total_params = sum(p.numel() for p in classifier.parameters() if
                               p.requires_grad)
    print("Total number of trainable parameters: ", pytorch_total_params)

    return detector, classifier


opt = parse_opts_online()

ROOT = os.path.dirname(__file__)

logger = logging.getLogger("pc")
pcs = set()
relay = MediaRelay()

__SENTINEL__ = "__SENTINEL__"
media_processing_thread_id_generator = itertools.count()

class VideoTransformTrack(MediaStreamTrack):
    """
    A video stream track that transforms frames from an another track.
    """

    kind = "video"

    def __int__(self):
        super().__init__()

    def prepare(self, track, transform, options):
        # super().__init__()  # don't forget this!
        self.track = track
        self.transform = transform
        self._thread = threading.Thread(target=self._run_worker_thread)
        self._in_queue = queue.Queue()
        self._latest_result_img_lock = threading.Lock()

        self._busy = False
        self._latest_result_img: Union[np.ndarray, None] = None

        self._thread.start()
        self.result = "init result"
        self._websocket = None
        self.ws_queue = queue.Queue()
        self.ws_thread = threading.Thread(target=self.start_ws)

        self.opt = options
        self.detector, self.classifier = load_models(self.opt)

        if self.opt.no_mean_norm and not self.opt.std_norm:
            norm_method = Normalize([0, 0, 0], [1, 1, 1])
        elif not self.opt.std_norm:
            norm_method = Normalize(self.opt.mean, [1, 1, 1])
        else:
            norm_method = Normalize(self.opt.mean, self.opt.std)

        self.spatial_transform = Compose([
            Scale(112),
            CenterCrop(112),
            ToTensor(self.opt.norm_value), norm_method
        ])

        self.opt.sample_duration = max(self.opt.sample_duration_clf, self.opt.sample_duration_det)
        cap = cv2.VideoCapture(self.opt.video)
        self.num_frame = 0
        self.clip = []
        self.active_index = 0
        self.passive_count = 0
        self.active = False
        self.prev_active = False
        self.finished_prediction = None
        self.pre_predict = False
        self.detector.eval()
        self.classifier.eval()
        self.cum_sum = np.zeros(self.opt.n_classes_clf, )
        self.clf_selected_queue = np.zeros(self.opt.n_classes_clf, )
        self.det_selected_queue = np.zeros(self.opt.n_classes_det, )
        self.myqueue_det = Queue(self.opt.det_queue_size, n_classes=self.opt.n_classes_det)
        self.myqueue_clf = Queue(self.opt.clf_queue_size, n_classes=self.opt.n_classes_clf)
        self.results = []
        self.prev_best1 = self.opt.n_classes_clf
        self.spatial_transform.randomize_parameters()
        self.x = []
        with open("annotation_EgoGesture/classIndAll.txt") as file:
            for l in file:
                self.x.append(l.strip())


    engaged = 0
    COLOCK = 0
    num = 1.0
    fps = ""

    def delay(self, frame):
        img = frame.to_ndarray(format="bgr24")

        frame_np = cv2.resize(img, (640, 480), cv2.INTER_AREA)
        frame_np = cv2.flip(frame_np, 1)
        frame_np = self.inference(frame_np)
        view_np = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)

        now = time.time()
        if now - self.COLOCK > 1:
            self.fps = " FPS:" + str(int(self.num / (now - self.COLOCK)))
            self.num = 1
            self.COLOCK = time.time()
        else:
            self.num += 1
        font = cv2.FONT_HERSHEY_COMPLEX
        cv2.putText(view_np,
                    time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()) + self.fps,
                    (0, 100),
                    font,
                    1,
                    (255, 5, 5),
                    3)

        new_frame = av.VideoFrame.from_ndarray(view_np, format="bgr24")
        new_frame.pts = frame.pts
        new_frame.time_base = frame.time_base

        return new_frame

    def inference(self, frame):
        t1 = time.time()
        # ret, frame = cap.read()
        if self.num_frame == 0:
            cur_frame = cv2.resize(frame, (320, 240))
            cur_frame = Image.fromarray(cv2.cvtColor(cur_frame, cv2.COLOR_BGR2RGB))
            cur_frame = cur_frame.convert('RGB')
            for i in range(self.opt.sample_duration):
                self.clip.append(cur_frame)
            self.clip = [self.spatial_transform(img) for img in self.clip]
        self.clip.pop(0)
        _frame = cv2.resize(frame, (320, 240))
        _frame = Image.fromarray(cv2.cvtColor(_frame, cv2.COLOR_BGR2RGB))
        _frame = _frame.convert('RGB')
        _frame = self.spatial_transform(_frame)
        self.clip.append(_frame)
        im_dim = self.clip[0].size()[-2:]
        try:
            test_data = torch.cat(self.clip, 0).view((self.opt.sample_duration, -1) + im_dim).permute(1, 0, 2, 3)
        except Exception as e:
            pdb.set_trace()
            raise e
        inputs = torch.cat([test_data], 0).view(1, 3, self.opt.sample_duration, 112, 112)
        self.num_frame += 1

        ground_truth_array = np.zeros(self.opt.n_classes_clf + 1, )
        with torch.no_grad():
            inputs = Variable(inputs)
            inputs_det = inputs[:, :, -self.opt.sample_duration_det:, :, :]
            outputs_det = self.detector(inputs_det)
            outputs_det = F.softmax(outputs_det, dim=1)
            outputs_det = outputs_det.cpu().numpy()[0].reshape(-1, )
            # enqueue the probabilities to the detector queue
            self.myqueue_det.enqueue(outputs_det.tolist())

            if self.opt.det_strategy == 'raw':
                det_selected_queue = outputs_det
            elif self.opt.det_strategy == 'median':
                det_selected_queue = self.myqueue_det.median
            elif self.opt.det_strategy == 'ma':
                det_selected_queue = self.myqueue_det.ma
            elif self.opt.det_strategy == 'ewma':
                det_selected_queue = self.myqueue_det.ewma
            prediction_det = np.argmax(det_selected_queue)

            prob_det = det_selected_queue[prediction_det]

            #### State of the detector is checked here as detector act as a switch for the classifier
            if prediction_det == 1:
                inputs_clf = inputs[:, :, :, :, :]
                inputs_clf = torch.Tensor(inputs_clf.numpy()[:, :, ::1, :, :])
                outputs_clf = self.classifier(inputs_clf)
                outputs_clf = F.softmax(outputs_clf, dim=1)
                outputs_clf = outputs_clf.cpu().numpy()[0].reshape(-1, )
                # Push the probabilities to queue
                self.myqueue_clf.enqueue(outputs_clf.tolist())
                self.passive_count = 0

                if self.opt.clf_strategy == 'raw':
                    clf_selected_queue = outputs_clf
                elif self.opt.clf_strategy == 'median':
                    clf_selected_queue = self.myqueue_clf.median
                elif self.opt.clf_strategy == 'ma':
                    clf_selected_queue = self.myqueue_clf.ma
                elif self.opt.clf_strategy == 'ewma':
                    clf_selected_queue = self.myqueue_clf.ewma

            else:
                outputs_clf = np.zeros(self.opt.n_classes_clf, )
                # Push the probabilities to queue
                self.myqueue_clf.enqueue(outputs_clf.tolist())
                self.passive_count += 1

        if self.passive_count >= self.opt.det_counter:
            active = False
        else:
            active = True

        # one of the following line need to be commented !!!!
        if active:
            self.active_index += 1
            self.cum_sum = ((self.cum_sum * (self.active_index - 1)) + (
                        weighting_func(self.active_index) * clf_selected_queue)) / self.active_index  # Weighted Aproach
            # self.cum_sum = ((self.cum_sum * (self.active_index-1)) + (1.0 * clf_selected_queue))/self.active_index #Not Weighting Aproach
            best2, best1 = tuple(self.cum_sum.argsort()[-2:][::1])
            if float(self.cum_sum[best1] - self.cum_sum[best2]) > self.opt.clf_threshold_pre:
                self.finished_prediction = True
                self.pre_predict = True

        else:
            self.active_index = 0
        if active == False and self.prev_active == True:
            self.finished_prediction = True
        elif active == True and self.prev_active == False:
            self.finished_prediction = False

        if self.finished_prediction:
            # print(finished_prediction,pre_predict)
            best2, best1 = tuple(self.cum_sum.argsort()[-2:][::1])
            if self.cum_sum[best1] > self.opt.clf_threshold_final:
                if self.pre_predict:
                    if best1 != self.prev_best1:
                        if self.cum_sum[best1] > self.opt.clf_threshold_final:
                            self.results.append(((i * self.opt.stride_len) + self.opt.sample_duration_clf, best1))
                            print('Early Detected - class : {} with prob : {} at frame {}'.format(best1, self.cum_sum[best1],
                                                                                                  (
                                                                                                          i * self.opt.stride_len) + self.opt.sample_duration_clf))
                else:
                    if self.cum_sum[best1] > self.opt.clf_threshold_final:
                        if best1 == self.prev_best1:
                            if self.cum_sum[best1] > 5:
                                self.results.append(((i * self.opt.stride_len) + self.opt.sample_duration_clf, best1))
                                print('Late Detected - class : {} with prob : {} at frame {}'.format(best1,
                                                                                                     self.cum_sum[best1], (
                                                                                                             i * self.opt.stride_len) + self.opt.sample_duration_clf))
                        else:
                            self.results.append(((i * self.opt.stride_len) + self.opt.sample_duration_clf, best1))

                            print('Late Detected - class : {} with prob : {} at frame {}'.format(best1, self.cum_sum[best1],
                                                                                                 (
                                                                                                         i * self.opt.stride_len) + self.opt.sample_duration_clf))

                self.finished_prediction = False
                self.prev_best1 = best1

            self.cum_sum = np.zeros(self.opt.n_classes_clf, )

        if active == False and self.prev_active == True:
            pre_predict = False

        self.prev_active = active
        elapsedTime = time.time() - t1
        fps = "(Playback) {:.1f} FPS".format(1 / elapsedTime)

        if len(self.results) != 0:
            predicted = np.array(self.results)[:, 1]
            self.prev_best1 = -1
        else:
            predicted = []

        if len(predicted) > 2:
            self.results = []
        for _, indx in enumerate(predicted):
            fps += " " + self.x[int(indx)]
        # print('predicted classes: \t', predicted)

        frame = cv2.resize(frame, (640, 480), cv2.INTER_AREA)
        cv2.putText(frame, fps, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (38, 0, 255), 1, cv2.LINE_AA)
        # cv2.imshow("Result", frame)

        # if cv2.waitKey(1)&0xFF == ord('q'):
        #     break
        return frame
    # cv2.destroyAllWindows()

    async def ws_sender(self, websocket, path):
        print("ws sender:", threading.currentThread())
        self._websocket = websocket
        if self._thread.is_alive() is False:
            try:
                self._thread.start()
            except Exception as exp:
                print("self._thread exception:", exp)
            time.sleep(1)

    def start_ws(self):
        new_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(new_loop)
        # start_server = websockets.serve(self.ws_sender, "0.0.0.0", 6789)
        # start_server.max_size = 2 ** 30
        # start_server.ping_interval = 5
        #
        # asyncio.get_event_loop().run_until_complete(start_server)
        # asyncio.get_event_loop().run_forever()

    def _start(self):
        if self._thread:
            return

        self._in_queue: queue.Queue = queue.LifoQueue()
        self._out_lock = threading.Lock()
        self._out_deque: deque = deque([])

        self._thread = threading.Thread(
            target=self._run_worker_thread,
            name=f"async_media_processor_{next(media_processing_thread_id_generator)}",
            daemon=True,
        )
        self._thread.start()

    def stop(self):
        self._in_queue.put(__SENTINEL__)
        self._thread.join(1.0)

        return super().stop()

    def _run_worker_thread(self):
        try:
            # self._worker_thread()
            asyncio.run(self._worker_thread())
        except Exception:
            logger.error("Error occurred in the WebRTC thread:")

            exc_type, exc_value, exc_traceback = sys.exc_info()
            for tb in traceback.format_exception(exc_type, exc_value, exc_traceback):
                for tbline in tb.rstrip().splitlines():
                    logger.error(tbline.rstrip())

    async def _worker_thread(self):
        while True:
            item = self._in_queue.get()
            if item == __SENTINEL__:
                break

            stop_requested = False
            while not self._in_queue.empty():
                item = self._in_queue.get_nowait()
                if item == __SENTINEL__:
                    stop_requested = True
            if stop_requested:
                break

            if item is None:
                raise Exception("A queued item is unexpectedly None")
            # 图像的耗时操作
            result_img = self.delay(item)

            with self._latest_result_img_lock:
                self.result = time.ctime()
                self.ws_queue.put("notify")
                self._latest_result_img = result_img

    async def recv(self):
        frame = await self.track.recv()
        self._in_queue.put(frame)

        with self._latest_result_img_lock:
            if self._latest_result_img is not None:

                return self._latest_result_img
            else:
                return frame


class ServerRTC(object):
    options = None

    async def index(self, request):
        content = open("index.html", "r").read()
        return web.Response(content_type="text/html", text=content)


    async def javascript(self, request):
        content = open("client.js", "r").read()
        return web.Response(content_type="application/javascript", text=content)


    async def offer(self, request):
        params = await request.json()
        offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

        pc = RTCPeerConnection()
        pc_id = "PeerConnection(%s)" % uuid.uuid4()
        pcs.add(pc)

        def log_info(msg, *args):
            logger.info(pc_id + " " + msg, *args)

        log_info("Created for %s", request.remote)

        # prepare local media
        player = MediaPlayer("demo-instruct.wav")
        if args.record_to:
            recorder = MediaRecorder(args.record_to)
        else:
            recorder = MediaBlackhole()

        origin_vtt = VideoTransformTrack()

        @pc.on("datachannel")
        def on_datachannel(channel):
            origin_vtt.data_channel = channel
            @channel.on("message")
            def on_message(message):
                if isinstance(message, str) and message.startswith("ping"):
                    channel.send(json.dumps({"tmps": int(time.time())}))

        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            log_info("Connection state is %s", pc.connectionState)
            if pc.connectionState == "failed":
                await pc.close()
                pcs.discard(pc)

        @pc.on("track")
        def on_track(track):
            log_info("Track %s received", track.kind)

            if track.kind == "audio":
                pc.addTrack(player.audio)
                recorder.addTrack(track)
            elif track.kind == "video":
                origin_vtt.prepare(track, params["video_transform"], self.options)
                pc.addTrack(origin_vtt)
                if args.record_to:
                    recorder.addTrack(relay.subscribe(track))

            @track.on("ended")
            async def on_ended():
                log_info("Track %s ended", track.kind)
                await recorder.stop()

        # handle offer
        await pc.setRemoteDescription(offer)
        await recorder.start()

        # send answer
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        return web.Response(
            content_type="application/json",
            text=json.dumps(
                {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
            ),
        )


    async def on_shutdown(self, app):
        # close peer connections
        coros = [pc.close() for pc in pcs]
        await asyncio.gather(*coros)
        pcs.clear()


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(
    #     description="WebRTC audio / video / data-channels demo"
    # )
    # parser.add_argument("--cert-file", help="SSL certificate file (for HTTPS)")
    # parser.add_argument("--key-file", help="SSL key file (for HTTPS)")
    # parser.add_argument(
    #     "--host", default="0.0.0.0", help="Host for HTTP server (default: 0.0.0.0)"
    # )
    # parser.add_argument(
    #     "--port", type=int, default=5000, help="Port for HTTP server (default: 8080)"
    # )
    # parser.add_argument("--record-to", help="Write received media to a file."),
    # parser.add_argument("--verbose", "-v", action="count")
    # args = parser.parse_args()
    #
    # if args.verbose:
    #     logging.basicConfig(level=logging.DEBUG)
    # else:
    #     logging.basicConfig(level=logging.INFO)
    #
    # if args.cert_file is None:
    #     ssl_context = ssl.SSLContext()
    #     ssl_context.load_cert_chain("example.crt", "example.key")
    # else:
    #     ssl_context = None

    logging.basicConfig(level=logging.DEBUG)
    ssl_context = ssl.SSLContext()
    ssl_context.load_cert_chain("example.crt", "example.key")

    sv = ServerRTC()
    sv.options = opt
    app = web.Application()
    app.on_shutdown.append(sv.on_shutdown)
    app.router.add_get("/", sv.index)
    app.router.add_get("/client.js", sv.javascript)
    app.router.add_post("/offer", sv.offer)
    web.run_app(
        app, access_log=None, host="0.0.0.0", port=5000, ssl_context=ssl_context
    )
