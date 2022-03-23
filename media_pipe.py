import argparse
import asyncio
import json
import logging
import os
import ssl
import uuid
import time
import cv2
from aiohttp import web
import av
import queue
from collections import deque
import threading
import itertools
from typing import Generic, List, Optional, Union
import traceback
import sys
import numpy as np

from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaBlackhole, MediaPlayer, MediaRecorder, MediaRelay

import socket
import websockets
from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import mediapipe as mp
import moosegesture

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

    def prepare(self, track, transform):
        # super().__init__()  # don't forget this!
        self.track = track
        self.transform = transform
        self._thread = threading.Thread(target=self._run_worker_thread)
        self._in_queue = queue.Queue()
        self._latest_result_img_lock = threading.Lock()

        self._busy = False
        self._latest_result_img: Union[np.ndarray, None] = None

        self.data_channel = None
        self._thread.start()
        self.result = "init result"
        self._websocket = None
        self.ws_queue = queue.Queue()
        self.ws_thread = threading.Thread(target=self.start_ws)

        self.mpHands = mp.solutions.hands
        self.hands = mp.solutions.hands.Hands()
        self.mpDraw = mp.solutions.drawing_utils


    engaged = 0
    COLOCK = 0
    num = 1.0
    fps = ""
    traces = [(320, 240)]*10
    hand_ges = "unknown"

    def delay(self, frame):
        img = frame.to_ndarray(format="bgr24")
        frame_np = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frame_np = cv2.resize(img, (640, 480), cv2.INTER_AREA)
        frame_np = cv2.flip(frame_np, 1)

        results = self.hands.process(frame_np)
        # print(results.multi_hand_landmarks)
        hand_points = []
        if results.multi_hand_landmarks:
            if len(results.multi_hand_landmarks) == 1:
                points = []
                for handLms in results.multi_hand_landmarks:
                    for id, lm in enumerate(handLms.landmark):
                        h, w, c = frame_np.shape
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        # if id == 8: # index finger tip
                        #     self.traces.append((cx, cy))
                        #     directions = moosegesture.getGesture(self.traces)
                        #     self.traces.pop(0)
                        #     if len(directions) == 1:
                        #         self.hand_ges = directions[0]
                        points.append((cx, cy))
                    hand_points.append(points)
                self.hand_ges = "one_hand"
            if len(results.multi_hand_landmarks) == 2:
                first_hand_mcp = (-10, -10)
                points = []
                for handLms in results.multi_hand_landmarks:
                    # handLMs are 21 points. so we need conection too-->mpHands.HAND_CONNECTIONS
                    for id, lm in enumerate(handLms.landmark):
                        # print(id, lm)
                        # lm = x,y cordinate of each landmark in float numbers. lm.x, lm.y methods
                        # So, need to covert in integer
                        h, w, c = frame_np.shape
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        # print(id, cx, cy)
                        points.append((cx, cy))
                        # if id == 5:  # (To draw 4th point)
                        #     cv2.circle(frame_np, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
                        #     if first_hand_mcp[0] < 0:
                        #         first_hand_mcp = (cx, cy)
                        #     else:
                        #         distance = abs(first_hand_mcp[0]-cx) + abs(first_hand_mcp[1]-cy)
                        #         if distance < 60:
                        #             self.hand_ges = "clasp_hand"
                    hand_points.append(points)
                    self.mpDraw.draw_landmarks(frame_np, handLms,
                                          self.mpHands.HAND_CONNECTIONS)  # drawing points and lines(=handconections)
                self.hand_ges = "two_hand"
        else:
            self.traces = [(320, 240)] * 10
            self.hand_ges = "unknown"

        view_np = frame_np
        now = time.time()
        if now - self.COLOCK > 1:
            self.fps = " FPS:" + str(int(self.num/(now - self.COLOCK)))
            self.num = 1
            self.COLOCK = time.time()
        else:
            self.num += 1
        font = cv2.FONT_HERSHEY_COMPLEX
        cv2.putText(view_np,
                    time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())+self.fps+" "+self.hand_ges,
                    (0, 50),
                    font,
                    1,
                    (255, 5, 5),
                    2)

        new_frame = av.VideoFrame.from_ndarray(view_np, format="bgr24")
        # new_frame.to_image().save("frame-{}.png".format(time.time_ns()))
        new_frame.pts = frame.pts
        new_frame.time_base = frame.time_base
        if self.data_channel is not None and self.data_channel.readyState == "open":
            result = {"timestamp": str(int(time.time())),
                      "code": 0,
                      "model": "hand_track",
                      "results":{"gesture":self.hand_ges,
                                 "points":hand_points}}
            if results.multi_hand_landmarks and len(results.multi_hand_landmarks) > 0:
                try:
                    self.data_channel.send(json.dumps(result))
                except Exception as exp:
                    print("delay send expt:", exp)
        return new_frame

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
        start_server = websockets.serve(self.ws_sender, "0.0.0.0", 6789)
        start_server.max_size = 2 ** 30
        start_server.ping_interval = 5

        asyncio.get_event_loop().run_until_complete(start_server)
        asyncio.get_event_loop().run_forever()

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
    async def index(self, request):
        content = open(os.path.join(ROOT, "index.html"), "r").read()
        return web.Response(content_type="text/html", text=content)


    async def javascript(self, request):
        content = open(os.path.join(ROOT, "client.js"), "r").read()
        return web.Response(content_type="application/javascript", text=content)


    response_json = {"timestamp": str(int(time.time())),
                      "code": 0,
                      "model": "hand_track",
                      "results":{"gesture":"unknown",
                                 "points":[]}}

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
        player = MediaPlayer(os.path.join(ROOT, "demo-instruct.wav"))

        recorder = MediaBlackhole()
        origin_vtt = VideoTransformTrack()

        @pc.on("datachannel")
        def on_datachannel(channel):
            origin_vtt.data_channel = channel
            @channel.on("message")
            def on_message(message):
                if isinstance(message, str) and message.startswith("ping"):
                    channel.send(json.dumps(self.response_json))

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
                origin_vtt.prepare(track, params["video_transform"])
                pc.addTrack(origin_vtt)

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
    parser = argparse.ArgumentParser(
        description="WebRTC audio / video / data-channels demo"
    )
    parser.add_argument("--cert-file", help="SSL certificate file (for HTTPS)")
    parser.add_argument("--key-file", help="SSL key file (for HTTPS)")
    parser.add_argument(
        "--host", default="0.0.0.0", help="Host for HTTP server (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", type=int, default=9003, help="Port for HTTP server (default: 8080)"
    )
    parser.add_argument("--record-to", help="Write received media to a file."),
    parser.add_argument("--verbose", "-v", action="count")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    if args.cert_file is None:
        ssl_context = ssl.SSLContext()
        ssl_context.load_cert_chain("example.crt", "example.key")
    else:
        ssl_context = None
    sv = ServerRTC()
    app = web.Application()
    app.on_shutdown.append(sv.on_shutdown)
    app.router.add_get("/", sv.index)
    app.router.add_get("/client.js", sv.javascript)
    app.router.add_post("/offer", sv.offer)
    web.run_app(
        app, access_log=None, host=args.host, port=args.port, ssl_context=ssl_context
    )
