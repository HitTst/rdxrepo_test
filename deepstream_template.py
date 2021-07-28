#!/usr/bin/env python3

################################################################################
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
################################################################################


from os.path import join
import gi
gi.require_version('Gst', '1.0')
from gi.repository import GObject, Gst
from gi.repository import GLib

import sys
sys.path.append('/opt/nvidia/deepstream/deepstream-5.0/lib')
import pyds

from ctypes import *
import configparser
import numpy as np
import datetime
import cv2
import time
import json
import math
import copy
import os

from common.is_aarch_64 import is_aarch64
from common.bus_call import bus_call
from common.utils import long_to_int

import logging
import sys

formatter = logging.Formatter('%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s')
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger = logging.getLogger("console_logger")
logger.setLevel(logging.DEBUG)
logger.addHandler(handler)

import socketio
from socketio.exceptions import *
sio = socketio.Client()

model_path = os.path.join(os.getcwd(), "models", "Primary_Detector")
labels_path = os.path.join(model_path, "labels.txt")
ip = os.environ.get('IP', '192.168.1.16') # update the ip of the socket server
port = os.environ.get('PORT', '5000') # update the port of the socket server


debug_info = {}
debug_config_path = os.path.join(os.getcwd(), "debug_config.json")
if os.path.exists(debug_config_path):
    with open(debug_config_path, "r") as f:
        debug_info = json.loads(f.read())

service_id = os.environ.get('SERVICE_ID', debug_info['service_id']) # change the service id of your choice
environment = os.environ.get('ENVIRONMENT', debug_info["environment"])

serverUrl = "http://{}:{}".format(ip, port)

uri_to_id_mapping = {}
obj_counter = {}
classes = []
count = 0
date = datetime.datetime.now().strftime("%d-%m-%Y")

buffer_counter = 0
image_buffer_cache_length = 10 # change image buffer size of your choice
image_buffer_cache = {}
image_buffer_cache_key = 0

loop = None
args = None
pipeline = None

start_time = time.time()

for key in range(0, image_buffer_cache_length):
    image_buffer_cache[str(key)] = {}

with open(labels_path, "r") as f:
    while True:
        line = f.readline()
        if not line:
            break
        classes.append(line.strip())
        obj_counter[count] = {'detections': []}
        count += 1

images_folder = os.path.join(os.getcwd(), "images")
if not os.path.exists(os.path.join(images_folder, date)):
    os.mkdir(os.path.join(images_folder, date))

from apscheduler.schedulers.background import BackgroundScheduler
sched = BackgroundScheduler()

def update_date():
    global date
    date = datetime.datetime.now().strftime("%d-%m-%Y")
    date_folder = os.path.join(images_folder, date)
    if not os.path.exists(date_folder):
        try:
            os.mkdir(date_folder)
        except:
            pass

sched.add_job(update_date, trigger='cron', hour='00', minute='00', second='01')
sched.start()

MUXER_OUTPUT_WIDTH=640
MUXER_OUTPUT_HEIGHT=480
MUXER_BATCH_TIMEOUT_USEC=4000000
TILED_OUTPUT_WIDTH=1280
TILED_OUTPUT_HEIGHT=720
GST_CAPS_FEATURES_NVMM="memory:NVMM"


@sio.event
def image(metadata):
    logger.debug(metadata)
    #convert python array into numpy array format.
    frame_image = np.array(
            image_buffer_cache[metadata["buffer_index"]][metadata["camera_id"]],
            copy=True, order='C')

    #covert the array into cv2 default color format
    frame_image = cv2.cvtColor(frame_image, cv2.COLOR_RGBA2BGR)
    image_path = os.path.join(images_folder, date, metadata["image_name"][-1])

    if len(metadata["image_name"]) == 2:
        prev_image_path = os.path.join(images_folder, date, metadata["image_name"][0])
        if not os.path.exists(prev_image_path):
            prev_date = (datetime.datetime.now()-datetime.timedelta(days=1)).strftime("%d-%m-%Y")
            prev_image_path = os.path.join(images_folder, prev_date, metadata["image_name"][0])
        prev_image = cv2.imread(prev_image_path) 
        frame_image = cv2.hconcat([prev_image, frame_image])

    if "bbox" in metadata.keys():
        frame_image = cv2.rectangle(frame_image,
                            (
                                metadata["bbox"]["left"], 
                                metadata["bbox"]["top"]
                            ),
                            (
                                metadata["bbox"]["left"] + metadata["bbox"]["width"], 
                                metadata["bbox"]["top"] + metadata["bbox"]["height"]
                            ),
                            (0,0,255,0),
                            2
                        )

    cv2.imwrite(image_path, frame_image)
    return None


@sio.event
def delete_image(metadata):
    image_path = os.path.join(images_folder, date, metadata["image_name"])
    if not os.path.exists(image_path):
            prev_date = (datetime.datetime.now()-datetime.timedelta(days=1)).strftime("%d-%m-%Y")
            image_path = os.path.join(images_folder, prev_date, metadata["image_name"])
    os.remove(image_path)
    return None


# tiler_sink_pad_buffer_probe  will extract metadata received on OSD sink pad
# and update params for drawing rectangle, object information etc.
def tiler_src_pad_buffer_probe(pad,info,u_data):
    global image_buffer_cache_key, buffer_counter, image_buffer_cache_length, image_buffer_cache, start_time
    object_detected = False
    camera_meatadata_dictionary = {}

    if time.time() - start_time > 1:
        image_buffer_cache_key = str(buffer_counter%image_buffer_cache_length)

        gst_buffer = info.get_buffer()
        if not gst_buffer:
            logger.debug("Unable to get GstBuffer ")
            return

        # Retrieve batch metadata from the gst_buffer
        # Note that pyds.gst_buffer_get_nvds_batch_meta() expects the
        # C address of gst_buffer as input, which is obtained with hash(gst_buffer)
        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
        l_frame = batch_meta.frame_meta_list

        while l_frame is not None:
            try:
                # Note that l_frame.data needs a cast to pyds.NvDsFrameMeta
                # The casting is done by pyds.NvDsFrameMeta.cast()
                # The casting also keeps ownership of the underlying memory
                # in the C code, so the Python garbage collector will leave
                # it alone.
                frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
            except StopIteration:
                break

            camera_id = list(uri_to_id_mapping.values())[frame_meta.source_id]
            camera_meatadata_dictionary[camera_id] = copy.deepcopy(obj_counter)
            camera_meatadata_dictionary[camera_id]['buffer_index'] = image_buffer_cache_key

            image_buffer_cache[image_buffer_cache_key].update({
                camera_id: pyds.get_nvds_buf_surface(hash(gst_buffer),frame_meta.batch_id)
            })

            l_obj=frame_meta.obj_meta_list

            while l_obj is not None:
                object_detected = True 

                try: 
                    # Casting l_obj.data to pyds.NvDsObjectMeta
                    obj_meta=pyds.NvDsObjectMeta.cast(l_obj.data)
                except StopIteration:
                    break
                
                camera_meatadata_dictionary[camera_id][obj_meta.class_id]['detections'].append({
                        'object_id': long_to_int(obj_meta.object_id),
                        'confidence': obj_meta.confidence,
                        'top': obj_meta.rect_params.top,
                        'left': obj_meta.rect_params.left,
                        'width': obj_meta.rect_params.width,
                        'height': obj_meta.rect_params.height
                    })
                

                try: 
                    l_obj=l_obj.next
                except StopIteration:
                    break      

            try:
                l_frame=l_frame.next
            except StopIteration:
                break

        if object_detected:
            if environment == "production":
                sio.emit('metadata', {"data": camera_meatadata_dictionary, "room": service_id})
            else:
                logger.debug(json.dumps({"data": camera_meatadata_dictionary, "room": service_id}, indent=4))
            if image_buffer_cache_key == str(image_buffer_cache_length-1):
                buffer_counter = 0
            else:
                buffer_counter += 1

        start_time = time.time()

    return Gst.PadProbeReturn.OK

def cb_newpad(decodebin, decoder_src_pad,data):
    logger.debug("In cb_newpad\n")
    caps=decoder_src_pad.get_current_caps()
    gststruct=caps.get_structure(0)
    gstname=gststruct.get_name()
    source_bin=data
    features=caps.get_features(0)

    # Need to check if the pad created by the decodebin is for video and not
    # audio.
    logger.debug("gstname={}".format(gstname))
    if(gstname.find("video")!=-1):
        # Link the decodebin pad only if decodebin has picked nvidia
        # decoder plugin nvdec_*. We do this by checking if the pad caps contain
        # NVMM memory features.
        logger.debug("features={}".format(features))
        if features.contains(GST_CAPS_FEATURES_NVMM):
            # Get the source bin ghost pad
            bin_ghost_pad=source_bin.get_static_pad("src")
            if not bin_ghost_pad.set_target(decoder_src_pad):
                sys.stderr.write("Failed to link decoder src pad to source bin ghost pad\n")
        else:
            sys.stderr.write(" Error: Decodebin did not pick nvidia decoder plugin.\n")

def decodebin_child_added(child_proxy,Object,name,user_data):
    logger.debug("Decodebin child added:{}\n".format(name))
    if(name.find("decodebin") != -1):
        Object.connect("child-added",decodebin_child_added,user_data)   
    if(is_aarch64() and name.find("nvv4l2decoder") != -1):
        logger.debug("Seting bufapi_version\n")
        Object.set_property("bufapi-version",True)

def create_source_bin(index,uri):
    logger.debug("Creating source bin")

    # Create a source GstBin to abstract this bin's content from the rest of the
    # pipeline
    bin_name="source-bin-%02d" %index
    logger.debug(bin_name)
    nbin=Gst.Bin.new(bin_name)
    if not nbin:
        sys.stderr.write(" Unable to create source bin \n")

    # Source element for reading from the uri.
    # We will use decodebin and let it figure out the container format of the
    # stream and the codec and plug the appropriate demux and decode plugins.
    uri_decode_bin=Gst.ElementFactory.make("uridecodebin", "uri-decode-bin")
    if not uri_decode_bin:
        sys.stderr.write(" Unable to create uri decode bin \n")
    # We set the input uri to the source element
    uri_decode_bin.set_property("uri",uri)
    # Connect to the "pad-added" signal of the decodebin which generates a
    # callback once a new pad for raw data has beed created by the decodebin
    uri_decode_bin.connect("pad-added",cb_newpad,nbin)
    uri_decode_bin.connect("child-added",decodebin_child_added,nbin)

    # We need to create a ghost pad for the source bin which will act as a proxy
    # for the video decoder src pad. The ghost pad will not have a target right
    # now. Once the decode bin creates the video decoder and generates the
    # cb_newpad callback, we will set the ghost pad target to the video decoder
    # src pad.
    Gst.Bin.add(nbin,uri_decode_bin)
    bin_pad=nbin.add_pad(Gst.GhostPad.new_no_target("src",Gst.PadDirection.SRC))
    if not bin_pad:
        sys.stderr.write(" Failed to add ghost pad in source bin \n")
        return None
    return nbin

def main(args):
    global loop, pipeline
    # Check input arguments
    if len(args) < 1:
        sys.stderr.write("usage: %s <uri1> [uri2] ... [uriN]\n" % args[0])
        sys.exit(1)

    number_sources=len(args)
    display = debug_info["display"]
    # Standard GStreamer initialization
    GObject.threads_init()
    Gst.init(None)

    # Create gstreamer elements */
    # Create Pipeline element that will form a connection of other elements
    logger.debug("Creating Pipeline \n ")
    pipeline = Gst.Pipeline()
    is_live = False

    if not pipeline:
        sys.stderr.write(" Unable to create Pipeline \n")
    logger.debug("Creating streamux \n ")

    # Create nvstreammux instance to form batches from one or more sources.
    streammux = Gst.ElementFactory.make("nvstreammux", "Stream-muxer")
    if not streammux:
        sys.stderr.write(" Unable to create NvStreamMux \n")

    pipeline.add(streammux)

    for i in range(number_sources):
        logger.debug("Creating source_bin {} \n ".format(i))
        uri_name=args[i]
        if uri_name.find("rtsp://") == 0 :
            is_live = True
        source_bin=create_source_bin(i, uri_name)
        if not source_bin:
            sys.stderr.write("Unable to create source bin \n")
        pipeline.add(source_bin)
        padname="sink_%u" %i
        sinkpad= streammux.get_request_pad(padname) 
        if not sinkpad:
            sys.stderr.write("Unable to create sink pad bin \n")
        srcpad=source_bin.get_static_pad("src")
        if not srcpad:
            sys.stderr.write("Unable to create src pad bin \n")
        srcpad.link(sinkpad)

    queue1=Gst.ElementFactory.make("queue","queue1")
    queue2=Gst.ElementFactory.make("queue","queue2")
    queue3=Gst.ElementFactory.make("queue","queue3")
    queue4=Gst.ElementFactory.make("queue","queue4")
    queue5=Gst.ElementFactory.make("queue","queue5")
    queue6=Gst.ElementFactory.make("queue","queue6")
    queue7=Gst.ElementFactory.make("queue","queue7")
    queue8=Gst.ElementFactory.make("queue","queue8")
    pipeline.add(queue1)
    pipeline.add(queue2)
    pipeline.add(queue3)
    pipeline.add(queue4)
    pipeline.add(queue5)
    pipeline.add(queue6)
    pipeline.add(queue7)
    pipeline.add(queue8)
    
    logger.debug("Creating Pgie \n ")
    pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
    if not pgie:
        sys.stderr.write(" Unable to create pgie \n")

    logger.debug("Creating Tracker \n ")
    tracker = Gst.ElementFactory.make("nvtracker", "tracker")
    if not tracker:
        sys.stderr.write(" Unable to create tracker \n")

    logger.debug("Creating nvvidconv1 \n ")
    nvvidconv1 = Gst.ElementFactory.make("nvvideoconvert", "convertor1")
    if not nvvidconv1:
        sys.stderr.write(" Unable to create nvvidconv1 \n")
    
    logger.debug("Creating filter1 \n ")
    caps1 = Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA")
    filter1 = Gst.ElementFactory.make("capsfilter", "filter1")
    if not filter1:
        sys.stderr.write(" Unable to get the caps filter1 \n")
    filter1.set_property("caps", caps1)

    logger.debug("Creating tiler \n ")
    tiler=Gst.ElementFactory.make("nvmultistreamtiler", "nvtiler")
    if not tiler:
        sys.stderr.write(" Unable to create tiler \n")

    logger.debug("Creating nvvidconv \n ")
    nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "convertor")
    if not nvvidconv:
        sys.stderr.write(" Unable to create nvvidconv \n")

    logger.debug("Creating nvosd \n ")
    nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")
    if not nvosd:
        sys.stderr.write(" Unable to create nvosd \n")

    if(is_aarch64()):
        logger.debug("Creating transform \n ")
        transform=Gst.ElementFactory.make("nvegltransform", "nvegl-transform")
        if not transform:
            sys.stderr.write(" Unable to create transform \n")

    logger.debug("Creating EGLSink \n")
    # sink = Gst.ElementFactory.make("nveglglessink", "nvvideo-renderer")
    if display:
        sink = Gst.ElementFactory.make("nveglglessink", "nvvideo-renderer")
    else:
        sink = Gst.ElementFactory.make("fakesink", "fakesink")
    if not sink:
        sys.stderr.write(" Unable to create egl sink \n")

    if is_live:
        logger.debug("Atleast one of the sources is live")
        streammux.set_property('live-source', 1)

    streammux.set_property('width', MUXER_OUTPUT_WIDTH)
    streammux.set_property('height', MUXER_OUTPUT_HEIGHT)
    streammux.set_property('batch-size', number_sources)
    streammux.set_property('batched-push-timeout', MUXER_BATCH_TIMEOUT_USEC)
    pgie.set_property('config-file-path', "dstestrdx_pgie_config.txt")
    pgie_batch_size=pgie.get_property("batch-size")
    if(pgie_batch_size != number_sources):
        logger.debug("WARNING: Overriding infer-config batch-size {} with number of sources {} \n".format(pgie_batch_size, number_sources))
        pgie.set_property("batch-size",number_sources)
    tiler_rows=int(math.sqrt(number_sources))
    tiler_columns=int(math.ceil((1.0*number_sources)/tiler_rows))
    tiler.set_property("rows",tiler_rows)
    tiler.set_property("columns",tiler_columns)
    tiler.set_property("width", TILED_OUTPUT_WIDTH)
    tiler.set_property("height", TILED_OUTPUT_HEIGHT)
    sink.set_property("qos",0)
    sink.set_property("sync",0)

    #Set properties of tracker
    config = configparser.ConfigParser()
    config.read('dstestrdx_tracker_config.txt')
    config.sections()

    for key in config['tracker']:
        if key == 'tracker-width' :
            tracker_width = config.getint('tracker', key)
            tracker.set_property('tracker-width', tracker_width)
        if key == 'tracker-height' :
            tracker_height = config.getint('tracker', key)
            tracker.set_property('tracker-height', tracker_height)
        if key == 'gpu-id' :
            tracker_gpu_id = config.getint('tracker', key)
            tracker.set_property('gpu_id', tracker_gpu_id)
        if key == 'll-lib-file' :
            tracker_ll_lib_file = config.get('tracker', key)
            tracker.set_property('ll-lib-file', tracker_ll_lib_file)
        if key == 'll-config-file' :
            tracker_ll_config_file = config.get('tracker', key)
            tracker.set_property('ll-config-file', tracker_ll_config_file)
        if key == 'enable-batch-process' :
            tracker_enable_batch_process = config.getint('tracker', key)
            tracker.set_property('enable_batch_process', tracker_enable_batch_process)

    logger.debug("Adding elements to Pipeline \n")
    pipeline.add(pgie)
    pipeline.add(tracker)
    pipeline.add(tiler)
    pipeline.add(nvvidconv)
    pipeline.add(filter1)
    pipeline.add(nvvidconv1)
    pipeline.add(nvosd)
    if is_aarch64():
        pipeline.add(transform)
    pipeline.add(sink)

    logger.debug("Linking elements in the Pipeline \n")
    streammux.link(queue1)
    queue1.link(pgie)
    pgie.link(queue2)
    queue2.link(tracker)
    tracker.link(queue3)
    queue3.link(nvvidconv1)
    nvvidconv1.link(queue4)
    queue4.link(filter1)
    filter1.link(queue5)
    queue5.link(tiler)
    tiler.link(queue6)
    queue6.link(nvvidconv)
    nvvidconv.link(queue7)
    queue7.link(nvosd)
    if is_aarch64() and display:
        nvosd.link(queue8)
        queue8.link(transform)
        transform.link(sink)
    else:
        nvosd.link(queue8)
        queue8.link(sink)    

    # create an event loop and feed gstreamer bus mesages to it
    loop = GObject.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect ("message", bus_call, loop)
    tiler_src_pad=tiler.get_static_pad("sink")
    if not tiler_src_pad:
        sys.stderr.write(" Unable to get src pad \n")

    tiler_src_pad.add_probe(Gst.PadProbeType.BUFFER, tiler_src_pad_buffer_probe, 0)

    # List the sources
    logger.debug("Now playing...")
    for i, source in enumerate(args):
        logger.debug("{}: {}".format(i, source))

    logger.debug("Starting pipeline \n")
    # start play back and listed to events		
    pipeline.set_state(Gst.State.PLAYING)
    try:
        loop.run()
    except:
        pass
    # cleanup
    logger.debug("Exiting app\n")
    pipeline.set_state(Gst.State.NULL)

@sio.on('stop')
def disconnect_from_server(id):
    if id == service_id:
        if loop:
            loop.quit()
            pipeline.set_state(Gst.State.NULL)
        sio.disconnect()

@sio.on('restart')
def restart_service():
    global uri_to_id_mapping, pipeline
    if loop: 
        loop.quit()
        pipeline.set_state(Gst.State.NULL)
    time.sleep(5)
    uri_to_id_mapping = {}
    sio.emit('info', service_id)


@sio.on('start')
def start_deepstream_app(metadata):
    logger.debug(metadata)
    global uri_to_id_mapping

    for camera, info in metadata.items():
        uri_to_id_mapping[metadata[camera]['link']] = camera

    if not len(uri_to_id_mapping.keys()):
        time.sleep(10)
        sio.emit('info', service_id)
        return True

    main(list(uri_to_id_mapping.keys()))
    logger.debug("shutting down")
    sio.disconnect()
    sys.exit()


if __name__ == '__main__':
    if environment == "production":
        while True:
            try:
                # connect to socket server
                sio.connect(serverUrl)
                break
            except ConnectionError:
                logger.debug("Connection error. Retrying ...!")
                time.sleep(10)
            except ConnectionRefusedError:
                logger.debug("Connection refused error. Retrying ...!")
                time.sleep(10)

        # join the room of the current service
        sio.emit('join', {'room': service_id})

        # call info channel to fetch the info
        sio.emit('info', service_id)

    elif environment == "test":

        for camera, info in debug_info["data"].items():
            uri_to_id_mapping[debug_info["data"][camera]['link']] = camera

        sys.exit(main(list(uri_to_id_mapping.keys())))
        
