/*
 * Copyright (c) 2018-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */
#include <gst/gst.h>
#include <glib.h>

#include <math.h>

#include <stdio.h>
#include <string.h>
#include "cuda_runtime_api.h"

#include <opencv2/objdetect/objdetect.hpp>

#include "gstnvdsmeta.h"
#include "gstnvdsinfer.h"
#include "nvdsinfer_custom_impl.h"
#include "nvds_version.h"

#define MAX_DISPLAY_LEN 64

#define TRACKER_CONFIG_FILE "/home/casit/gstreamerTest/dstest2_tracker_config.txt"
#define MAX_TRACKING_ID_LEN 16

#define PGIE_CLASS_ID_VEHICLE 1
#define PGIE_CLASS_ID_PERSON 0

/* The muxer output resolution must be set if the input streams will be of
 * different resolution. The muxer will scale all the input frames to this
 * resolution. */
#define MUXER_OUTPUT_WIDTH 1920
#define MUXER_OUTPUT_HEIGHT 1080

#define PGIE_DETECTED_CLASS_NUM 4

#define PGIE_NET_WIDTH 960
#define PGIE_NET_HEIGHT 544

/* Muxer batch formation timeout, for e.g. 40 millisec. Should ideally be set
 * based on the fastest source's framerate. */
#define MUXER_BATCH_TIMEOUT_USEC 40000

gint frame_number = 0;
gint sgie_number = 0;
/* These are the strings of the labels for the respective models */
const gchar sgie1_classes_str[2][32] = {
        "normal", "fall"
};

const gchar sgie2_classes_str[20][32] = {
        "Acura", "Audi", "BMW", "Chevrolet", "Chrysler",
        "Dodge", "Ford", "GMC", "Honda", "Hyundai", "Infiniti", "Jeep", "Kia",
        "Lexus", "Mazda", "Mercedes", "Nissan",
        "Subaru", "Toyota", "Volkswagen"
};

const gchar sgie3_classes_str[6][32] = {
        "coupe", "largevehicle", "sedan", "suv",
        "truck", "van"
};

const gchar pgie_classes_str[PGIE_DETECTED_CLASS_NUM][32] =
        {"Person", "helmet", "cellphone", "belt"};

/* gie_unique_id is one of the properties in the above dstensor_sgiex_config.txt
 * files. These should be unique and known when we want to parse the Metadata
 * respective to the sgie labels. Ideally these should be read from the config
 * files but for brevity we ensure they are same. */

const guint sgie1_unique_id = 2;
const guint sgie2_unique_id = 3;
const guint sgie3_unique_id = 4;

/* nvds_lib_major_version and nvds_lib_minor_version is the version number of
 * deepstream sdk */

unsigned int nvds_lib_major_version = NVDS_VERSION_MAJOR;
unsigned int nvds_lib_minor_version = NVDS_VERSION_MINOR;

static GstPadProbeReturn
osd_sink_pad_buffer_probe (GstPad * pad, GstPadProbeInfo * info,
                           gpointer u_data)
{
    GstBuffer *buf = (GstBuffer *) info->data;
    guint num_rects = 0;
    NvDsObjectMeta *obj_meta = NULL;
    guint vehicle_count = 0;
    guint person_count = 0;
    NvDsMetaList * l_frame = NULL;
    NvDsMetaList * l_obj = NULL;
    NvDsDisplayMeta *display_meta = NULL;

    NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta (buf);

    for (l_frame = batch_meta->frame_meta_list; l_frame != NULL;
         l_frame = l_frame->next) {
        NvDsFrameMeta *frame_meta = (NvDsFrameMeta *) (l_frame->data);
        int offset = 0;
        for (l_obj = frame_meta->obj_meta_list; l_obj != NULL;
             l_obj = l_obj->next) {
            obj_meta = (NvDsObjectMeta *) (l_obj->data);
            if (obj_meta->class_id == PGIE_CLASS_ID_VEHICLE) {
                vehicle_count++;
                num_rects++;
            }
            if (obj_meta->class_id == PGIE_CLASS_ID_PERSON) {
                person_count++;
                num_rects++;
            }
            if (obj_meta->class_id == 2) {
                g_print("cellphone detect");
            }
        }
        display_meta = nvds_acquire_display_meta_from_pool(batch_meta);
        NvOSD_TextParams *txt_params  = &display_meta->text_params[0];
        display_meta->num_labels = 1;
        txt_params->display_text = g_malloc0 (MAX_DISPLAY_LEN);
        offset = snprintf(txt_params->display_text, MAX_DISPLAY_LEN, "Person = %d ", person_count);
        offset = snprintf(txt_params->display_text + offset , MAX_DISPLAY_LEN, "Vehicle = %d ", vehicle_count);

        /* Now set the offsets where the string should appear */
        txt_params->x_offset = 10;
        txt_params->y_offset = 12;

        /* Font , font-color and font-size */
        txt_params->font_params.font_name = "Serif";
        txt_params->font_params.font_size = 10;
        txt_params->font_params.font_color.red = 1.0;
        txt_params->font_params.font_color.green = 1.0;
        txt_params->font_params.font_color.blue = 1.0;
        txt_params->font_params.font_color.alpha = 1.0;

        /* Text background color */
        txt_params->set_bg_clr = 1;
        txt_params->text_bg_clr.red = 0.0;
        txt_params->text_bg_clr.green = 0.0;
        txt_params->text_bg_clr.blue = 0.0;
        txt_params->text_bg_clr.alpha = 1.0;

        nvds_add_display_meta_to_frame(frame_meta, display_meta);
    }

//    g_print ("Frame Number = %d Number of objects = %d "
//             "Helmet Count = %d Person Count = %d\n",
//             frame_number, num_rects, vehicle_count, person_count);
    frame_number++;
    return GST_PAD_PROBE_OK;
}

static GstPadProbeReturn
pgie_pad_buffer_probe (GstPad * pad, GstPadProbeInfo * info, gpointer u_data)
{
    static guint use_device_mem = 0;
    static NvDsInferNetworkInfo networkInfo
            {
                    PGIE_NET_WIDTH, PGIE_NET_HEIGHT, 3};
    NvDsInferParseDetectionParams detectionParams;
    detectionParams.numClassesConfigured = 4;
    detectionParams.perClassPreclusterThreshold = {0.2, 0.2, 0.2, 0.2};
    static float groupThreshold = 1;
    static float groupEps = 0.2;

//    NvDsBatchMeta *batch_meta =
//            gst_buffer_get_nvds_batch_meta (GST_BUFFER (info->data));

    GstBuffer *buf = (GstBuffer *) info->data;
    NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta (buf);

    /* Iterate each frame metadata in batch */
    for (NvDsMetaList * l_frame = batch_meta->frame_meta_list; l_frame != NULL;
         l_frame = l_frame->next) {
        NvDsFrameMeta *frame_meta = (NvDsFrameMeta *) l_frame->data;

        /* Iterate user metadata in frames to search PGIE's tensor metadata */
        for (NvDsMetaList * l_user = frame_meta->frame_user_meta_list;
             l_user != NULL; l_user = l_user->next) {
            NvDsUserMeta *user_meta = (NvDsUserMeta *) l_user->data;
            if (user_meta->base_meta.meta_type != NVDSINFER_TENSOR_OUTPUT_META)
                continue;

            /* convert to tensor metadata */
            NvDsInferTensorMeta *meta =
                    (NvDsInferTensorMeta *) user_meta->user_meta_data;
            for (unsigned int i = 0; i < meta->num_output_layers; i++) {
                NvDsInferLayerInfo *info = &meta->output_layers_info[i];
                info->buffer = meta->out_buf_ptrs_host[i];
                if (use_device_mem && meta->out_buf_ptrs_dev[i]) {
                    cudaMemcpy (meta->out_buf_ptrs_host[i], meta->out_buf_ptrs_dev[i],
                                info->inferDims.numElements * 4, cudaMemcpyDeviceToHost);
                }
            }
            /* Parse output tensor and fill detection results into objectList. */
            std::vector < NvDsInferLayerInfo >
                    outputLayersInfo (meta->output_layers_info,
                                      meta->output_layers_info + meta->num_output_layers);
            std::vector < NvDsInferObjectDetectionInfo > objectList;
#if NVDS_VERSION_MAJOR >= 5
            if (nvds_lib_major_version >= 5) {
        if (meta->network_info.width != networkInfo.width ||
            meta->network_info.height != networkInfo.height ||
            meta->network_info.channels != networkInfo.channels) {
          g_error ("failed to check pgie network info\n");
        }
      }
#endif
//            NvDsInferParseCustomResnet (outputLayersInfo, networkInfo,
//                                        detectionParams, objectList);

            /* Seperate detection rectangles per class for grouping. */
            std::vector < std::vector <
                    cv::Rect >> objectListClasses (PGIE_DETECTED_CLASS_NUM);
            for (auto & obj:objectList) {
                objectListClasses[obj.classId].emplace_back (obj.left, obj.top,
                                                             obj.width, obj.height);
            }

            for (uint32_t c = 0; c < objectListClasses.size (); ++c) {
                auto & objlist = objectListClasses[c];
                if (objlist.empty ())
                    continue;

                /* Merge and cluster similar detection results */
                cv::groupRectangles (objlist, groupThreshold, groupEps);

                /* Iterate final rectangules and attach result into frame's obj_meta_list. */
                for (const auto & rect:objlist) {
                    NvDsObjectMeta *obj_meta =
                            nvds_acquire_obj_meta_from_pool (batch_meta);
                    obj_meta->unique_component_id = meta->unique_id;
                    obj_meta->confidence = 0.0;

                    /* This is an untracked object. Set tracking_id to -1. */
                    obj_meta->object_id = UNTRACKED_OBJECT_ID;
                    obj_meta->class_id = c;

                    NvOSD_RectParams & rect_params = obj_meta->rect_params;
                    NvOSD_TextParams & text_params = obj_meta->text_params;

                    /* Assign bounding box coordinates. */
                    rect_params.left = rect.x * MUXER_OUTPUT_WIDTH / PGIE_NET_WIDTH;
                    rect_params.top = rect.y * MUXER_OUTPUT_HEIGHT / PGIE_NET_HEIGHT;
                    rect_params.width = rect.width * MUXER_OUTPUT_WIDTH / PGIE_NET_WIDTH;
                    rect_params.height =
                            rect.height * MUXER_OUTPUT_HEIGHT / PGIE_NET_HEIGHT;

                    /* Border of width 3. */
                    rect_params.border_width = 3;
                    rect_params.has_bg_color = 0;
                    rect_params.border_color = (NvOSD_ColorParams) {
                            1, 0, 0, 1};

                    /* display_text requires heap allocated memory. */
                    text_params.display_text = g_strdup (pgie_classes_str[c]);
                    /* Display text above the left top corner of the object. */
                    text_params.x_offset = rect_params.left;
                    text_params.y_offset = rect_params.top - 10;
                    /* Set black background for the text. */
                    text_params.set_bg_clr = 1;
                    text_params.text_bg_clr = (NvOSD_ColorParams) {
                            0, 0, 0, 1};
                    /* Font face, size and color. */
                    text_params.font_params.font_name = (gchar *) "Serif";
                    text_params.font_params.font_size = 11;
                    text_params.font_params.font_color = (NvOSD_ColorParams) {
                            1, 1, 1, 1};
//                    nvds_add_obj_meta_to_frame (frame_meta, obj_meta, NULL);
                }
            }
        }
    }
    use_device_mem = 1 - use_device_mem;
    return GST_PAD_PROBE_OK;
}

/* This is the buffer probe function that we have registered on the sink pad
 * of the tiler element. All SGIE infer elements in the pipeline shall attach
 * their NvDsInferTensorMeta to each object's metadata of each frame, here we will
 * iterate & parse the tensor data to get classification confidence and labels.
 * The result would be attached as classifier_meta into its object's metadata.
 */
static GstPadProbeReturn
sgie_pad_buffer_probe (GstPad * pad, GstPadProbeInfo * info, gpointer u_data)
{
    static guint use_device_mem = 0;

    NvDsBatchMeta *batch_meta =
            gst_buffer_get_nvds_batch_meta (GST_BUFFER (info->data));

    /* Iterate each frame metadata in batch */
    for (NvDsMetaList * l_frame = batch_meta->frame_meta_list; l_frame != NULL;
         l_frame = l_frame->next) {
        NvDsFrameMeta *frame_meta = (NvDsFrameMeta *) l_frame->data;

        /* Iterate object metadata in frame */
        for (NvDsMetaList * l_obj = frame_meta->obj_meta_list; l_obj != NULL;
             l_obj = l_obj->next) {
            NvDsObjectMeta *obj_meta = (NvDsObjectMeta *) l_obj->data;

            if (!obj_meta->obj_user_meta_list)
            {
                g_print ("obj meta none\n");
            } else g_print ("obj meta access\n");



            /* Iterate user metadata in object to search SGIE's tensor data */
            for (NvDsMetaList * l_user = obj_meta->obj_user_meta_list; l_user != NULL;
                 l_user = l_user->next) {

//                g_print ("obj nmeta noone\n");
                NvDsUserMeta *user_meta = (NvDsUserMeta *) l_user->data;
                if (user_meta->base_meta.meta_type != NVDSINFER_TENSOR_OUTPUT_META)
                    continue;

                /* convert to tensor metadata */
                NvDsInferTensorMeta *meta =
                        (NvDsInferTensorMeta *) user_meta->user_meta_data;

                for (unsigned int i = 0; i < meta->num_output_layers; i++) {
                    NvDsInferLayerInfo *info = &meta->output_layers_info[i];
                    info->buffer = meta->out_buf_ptrs_host[i];
                    if (use_device_mem && meta->out_buf_ptrs_dev[i]) {
                        cudaMemcpy (meta->out_buf_ptrs_host[i], meta->out_buf_ptrs_dev[i],
                                    info->inferDims.numElements * 4, cudaMemcpyDeviceToHost);
                    }
                }

                NvDsInferDimsCHW dims;

                getDimsCHWFromDims (dims, meta->output_layers_info[0].inferDims);
                unsigned int numClasses = dims.c;
                float *outputCoverageBuffer =
                        (float *) meta->output_layers_info[0].buffer;
                float maxProbability = 0;
                bool attrFound = false;
                NvDsInferAttribute attr;

                /* Iterate through all the probabilities that the object belongs to
                 * each class. Find the maximum probability and the corresponding class
                 * which meets the minimum threshold. */
                float total = 0.0;
                for (unsigned int c = 0; c < numClasses; c++) {
                    float probability = outputCoverageBuffer[c];
                    total += probability;
                    if (probability > 0.51 && probability > maxProbability) {
                        maxProbability = probability;
                        attrFound = true;
                        attr.attributeIndex = 0;
                        attr.attributeValue = c;
                        attr.attributeConfidence = probability;
                    }
                }
                g_print ("Frame Number = %d softmax output = normal: %f, fall: %f\n",
                         sgie_number, outputCoverageBuffer[0], outputCoverageBuffer[1]);

                /* Generate classifer metadata and attach to obj_meta */
                if (attrFound) {
                    NvDsClassifierMeta *classifier_meta =
                            nvds_acquire_classifier_meta_from_pool (batch_meta);

                    classifier_meta->unique_component_id = meta->unique_id;

                    NvDsLabelInfo *label_info =
                            nvds_acquire_label_info_meta_from_pool (batch_meta);
                    label_info->result_class_id = attr.attributeValue;
                    label_info->result_prob = attr.attributeConfidence;

                    /* Fill label name */
                    switch (meta->unique_id) {
                        case sgie1_unique_id:
                            strcpy (label_info->result_label,
                                    sgie1_classes_str[label_info->result_class_id]);
                            break;
                        case sgie2_unique_id:
                            strcpy (label_info->result_label,
                                    sgie2_classes_str[label_info->result_class_id]);
                            break;
                        case sgie3_unique_id:
                            strcpy (label_info->result_label,
                                    sgie3_classes_str[label_info->result_class_id]);
                            break;
                        default:
                            break;
                    }
                    gchar *temp = obj_meta->text_params.display_text;
                    obj_meta->text_params.display_text =
                            g_strconcat (temp, " ", label_info->result_label, nullptr);
                    g_free (temp);

                    nvds_add_label_info_meta_to_classifier (classifier_meta, label_info);
                    nvds_add_classifier_meta_to_object (obj_meta, classifier_meta);
                }
            }
        }
    }
    sgie_number++;
    use_device_mem = 1 - use_device_mem;
    return GST_PAD_PROBE_OK;
}

static gboolean
bus_call (GstBus * bus, GstMessage * msg, gpointer data)
{
    GMainLoop *loop = (GMainLoop *) data;
    switch (GST_MESSAGE_TYPE (msg)) {
        case GST_MESSAGE_EOS:
            g_print ("End of stream\n");
            g_main_loop_quit (loop);
            break;
        case GST_MESSAGE_ERROR:{
            gchar *debug;
            GError *error;
            gst_message_parse_error (msg, &error, &debug);
            g_printerr ("ERROR from element %s: %s\n",
                        GST_OBJECT_NAME (msg->src), error->message);
            if (debug)
                g_printerr ("Error details: %s\n", debug);
            g_free (debug);
            g_error_free (error);
            g_main_loop_quit (loop);
            break;
        }
        default:
            break;
    }
    return TRUE;
}

#define CONFIG_GROUP_TRACKER "tracker"
#define CONFIG_GROUP_TRACKER_WIDTH "tracker-width"
#define CONFIG_GROUP_TRACKER_HEIGHT "tracker-height"
#define CONFIG_GROUP_TRACKER_LL_CONFIG_FILE "ll-config-file"
#define CONFIG_GROUP_TRACKER_LL_LIB_FILE "ll-lib-file"
#define CONFIG_GROUP_TRACKER_ENABLE_BATCH_PROCESS "enable-batch-process"
#define CONFIG_GPU_ID "gpu-id"

#define CHECK_ERROR(error) \
    if (error) { \
        g_printerr ("Error while parsing config file: %s\n", error->message); \
        goto done; \
    }

static gchar *
get_absolute_file_path (gchar *cfg_file_path, gchar *file_path)
{
    gchar abs_cfg_path[PATH_MAX + 1];
    gchar *abs_file_path;
    gchar *delim;

    if (file_path && file_path[0] == '/') {
        return file_path;
    }

    if (!realpath (cfg_file_path, abs_cfg_path)) {
        g_free (file_path);
        return NULL;
    }

    // Return absolute path of config file if file_path is NULL.
    if (!file_path) {
        abs_file_path = g_strdup (abs_cfg_path);
        return abs_file_path;
    }

    delim = g_strrstr (abs_cfg_path, "/");
    *(delim + 1) = '\0';

    abs_file_path = g_strconcat (abs_cfg_path, file_path, NULL);
    g_free (file_path);

    return abs_file_path;
}


static gboolean
set_tracker_properties (GstElement *nvtracker)
{
    gboolean ret = FALSE;
    GError *error = NULL;
    gchar **keys = NULL;
    gchar **key = NULL;
    GKeyFile *key_file = g_key_file_new ();

    if (!g_key_file_load_from_file (key_file, TRACKER_CONFIG_FILE, G_KEY_FILE_NONE,
                                    &error)) {
        g_printerr ("Failed to load config file: %s\n", error->message);
        return FALSE;
    }

    keys = g_key_file_get_keys (key_file, CONFIG_GROUP_TRACKER, NULL, &error);
    CHECK_ERROR (error);

    for (key = keys; *key; key++) {
        if (!g_strcmp0 (*key, CONFIG_GROUP_TRACKER_WIDTH)) {
            gint width =
                    g_key_file_get_integer (key_file, CONFIG_GROUP_TRACKER,
                                            CONFIG_GROUP_TRACKER_WIDTH, &error);
            CHECK_ERROR (error);
            g_object_set (G_OBJECT (nvtracker), "tracker-width", width, NULL);
        } else if (!g_strcmp0 (*key, CONFIG_GROUP_TRACKER_HEIGHT)) {
            gint height =
                    g_key_file_get_integer (key_file, CONFIG_GROUP_TRACKER,
                                            CONFIG_GROUP_TRACKER_HEIGHT, &error);
            CHECK_ERROR (error);
            g_object_set (G_OBJECT (nvtracker), "tracker-height", height, NULL);
        } else if (!g_strcmp0 (*key, CONFIG_GPU_ID)) {
            guint gpu_id =
                    g_key_file_get_integer (key_file, CONFIG_GROUP_TRACKER,
                                            CONFIG_GPU_ID, &error);
            CHECK_ERROR (error);
            g_object_set (G_OBJECT (nvtracker), "gpu_id", gpu_id, NULL);
        } else if (!g_strcmp0 (*key, CONFIG_GROUP_TRACKER_LL_CONFIG_FILE)) {
            char* ll_config_file = get_absolute_file_path (TRACKER_CONFIG_FILE,
                                                           g_key_file_get_string (key_file,
                                                                                  CONFIG_GROUP_TRACKER,
                                                                                  CONFIG_GROUP_TRACKER_LL_CONFIG_FILE, &error));
            CHECK_ERROR (error);
            g_object_set (G_OBJECT (nvtracker), "ll-config-file", ll_config_file, NULL);
        } else if (!g_strcmp0 (*key, CONFIG_GROUP_TRACKER_LL_LIB_FILE)) {
            char* ll_lib_file = get_absolute_file_path (TRACKER_CONFIG_FILE,
                                                        g_key_file_get_string (key_file,
                                                                               CONFIG_GROUP_TRACKER,
                                                                               CONFIG_GROUP_TRACKER_LL_LIB_FILE, &error));
            CHECK_ERROR (error);
            g_object_set (G_OBJECT (nvtracker), "ll-lib-file", ll_lib_file, NULL);
        } else if (!g_strcmp0 (*key, CONFIG_GROUP_TRACKER_ENABLE_BATCH_PROCESS)) {
            gboolean enable_batch_process =
                    g_key_file_get_integer (key_file, CONFIG_GROUP_TRACKER,
                                            CONFIG_GROUP_TRACKER_ENABLE_BATCH_PROCESS, &error);
            CHECK_ERROR (error);
            g_object_set (G_OBJECT (nvtracker), "enable_batch_process",
                          enable_batch_process, NULL);
        } else {
            g_printerr ("Unknown key '%s' for group [%s]", *key,
                        CONFIG_GROUP_TRACKER);
        }
    }

    ret = TRUE;
    done:
    if (error) {
        g_error_free (error);
    }
    if (keys) {
        g_strfreev (keys);
    }
    if (!ret) {
        g_printerr ("%s failed", __func__);
    }
    return ret;
}

int main ()
{
    GMainLoop *loop = NULL;
    GstElement *pipeline = NULL, *source = NULL, *h264parser = NULL, *queue = NULL,
            *decoder = NULL, *streammux = NULL, *sink = NULL, *pgie = NULL, *nvvidconv = NULL,
            *nvosd = NULL, *sgie1 = NULL, *nvtracker = NULL, *tiler =
            NULL, *queue2, *queue3, *queue4, *queue5, *queue6;;
//    GstElement *pipeline = NULL, *source = NULL, *h264parser = NULL, *queue =
//            NULL, *decoder = NULL, *streammux = NULL, *sink = NULL, *pgie =
//            NULL, *nvvidconv = NULL, *nvosd = NULL, *sgie1 = NULL, *sgie2 =
//            NULL, *sgie3 = NULL, *tiler =
//            NULL, *queue2, *queue3, *queue4, *queue5, *queue6;
#ifdef PLATFORM_TEGRA
    GstElement *transform = NULL;
#endif
    GstBus *bus = NULL;
    guint bus_watch_id;
    GstPad *osd_sink_pad = NULL, *queue_src_pad = NULL, *tiler_sink_pad = NULL;

    /* Standard GStreamer initialization */
    gst_init (NULL, NULL);
    loop = g_main_loop_new (NULL, FALSE);

    /* Create gstreamer elements */
    /* Create Pipeline element that will form a connection of other elements */
    pipeline = gst_pipeline_new ("dstest1-pipeline");

    /* Source element for reading from the file */
    source = gst_element_factory_make ("filesrc", "file-source");

    /* Since the data format in the input file is elementary h264 stream,
     * we need a h264parser */
    h264parser = gst_element_factory_make ("h264parse", "h264-parser");
//    h264parser = gst_element_factory_make ("uridecodebin", "h264-parser");

    /* Use nvdec_h264 for hardware accelerated decode on GPU */
    decoder = gst_element_factory_make ("nvv4l2decoder", "nvv4l2-decoder");

    /* Create nvstreammux instance to form batches from one or more sources. */
    streammux = gst_element_factory_make ("nvstreammux", "stream-muxer");

    if (!pipeline || !streammux) {
        g_printerr ("One element could not be created. Exiting.\n");
        return -1;
    }

    /* Use nvinfer to run inferencing on decoder's output,
     * behaviour of inferencing is set through config file */
    pgie = gst_element_factory_make ("nvinfer", "primary-nvinference-engine");

    queue = gst_element_factory_make ("queue", NULL);
    queue2 = gst_element_factory_make ("queue", NULL);
    queue3 = gst_element_factory_make ("queue", NULL);
    queue4 = gst_element_factory_make ("queue", NULL);
    queue5 = gst_element_factory_make ("queue", NULL);
    queue6 = gst_element_factory_make ("queue", NULL);

    nvtracker = gst_element_factory_make ("nvtracker", "tracker");

    sgie1 = gst_element_factory_make ("nvinfer", "secondary1-nvinference-engine");

    /* Use convertor to convert from NV12 to RGBA as required by nvosd */
//    tiler = gst_element_factory_make ("nvmultistreamtiler", "tiler");

    /* Use convertor to convert from NV12 to RGBA as required by nvosd */
    nvvidconv = gst_element_factory_make ("nvvideoconvert", "nvvideo-converter");

    /* Create OSD to draw on the converted RGBA buffer */
    nvosd = gst_element_factory_make ("nvdsosd", "nv-onscreendisplay");

    /* Finally render the osd output */
#ifdef PLATFORM_TEGRA
    transform = gst_element_factory_make ("nvegltransform", "nvegl-transform");
#endif
    sink = gst_element_factory_make ("nveglglessink", "nvvideo-renderer");

    if (!source || !h264parser || !decoder || !pgie || !nvtracker ||
        !sgie1 || !nvvidconv || !nvosd || !sink) {
        g_printerr ("One element could not be created. Exiting.\n");
        return -1;
    }

#ifdef PLATFORM_TEGRA
    if(!transform) {
    g_printerr ("One tegra element could not be created. Exiting.\n");
    return -1;
  }
#endif

//    gchar* filePath = "/home/casit/gstreamerTest/sample_720p.h264";
//    gchar* filePath = "/home/casit/gstreamerTest/test.mp4";
    gchar* filePath = "/home/casit/gstreamerTest/test2.h264";
//    gchar* filePath = "/home/casit/gstreamerTest/testfall.h264";
    /* we set the input filename to the source element */
    g_object_set (G_OBJECT (source), "location", filePath, NULL);

    g_object_set (G_OBJECT (streammux), "batch-size", 1, NULL);

    g_object_set (G_OBJECT (streammux), "width", MUXER_OUTPUT_WIDTH, "height",
                  MUXER_OUTPUT_HEIGHT,
                  "batched-push-timeout", MUXER_BATCH_TIMEOUT_USEC, NULL);

    /* Set all the necessary properties of the nvinfer element,
     * the necessary ones are : */
//    gchar * pgiePath = "/home/casit/gstreamerTest/dstest1_pgie_config.txt";
    gchar * pgiePath = "/home/casit/gstreamerTest/dsdetect_pgie_config.txt";
    gchar * sgiePath = "/home/casit/gstreamerTest/dsclassification_sgie_config.txt";
//    g_object_set (G_OBJECT (pgie),
//                  "config-file-path", pgiePath, NULL);
//    g_object_set (G_OBJECT (sgie1),
//                  "config-file-path", sgiePath, NULL);

    g_object_set (G_OBJECT (pgie), "config-file-path", pgiePath,
                  "output-tensor-meta", TRUE, "batch-size", 1, NULL);
    g_object_set (G_OBJECT (sgie1), "config-file-path", sgiePath,
                  "output-tensor-meta", TRUE, "process-mode", 2, NULL);

    /* Set necessary properties of the tracker element. */
    if (!set_tracker_properties(nvtracker)) {
        g_printerr ("Failed to set tracker properties. Exiting.\n");
        return -1;
    }

    /* we add a message handler */
    bus = gst_pipeline_get_bus (GST_PIPELINE (pipeline));
    bus_watch_id = gst_bus_add_watch (bus, bus_call, loop);
    gst_object_unref (bus);

    /* Set up the pipeline */
    /* we add all elements into the pipeline */
#ifdef PLATFORM_TEGRA
    gst_bin_add_many (GST_BIN (pipeline),
      source, h264parser, decoder, streammux, pgie,
      nvvidconv, nvosd, transform, sink, NULL);
#else
    gst_bin_add_many (GST_BIN (pipeline),
                      source, h264parser, decoder, streammux, pgie, queue, nvtracker, sgie1, queue5,
                      nvvidconv, nvosd, sink, NULL);

#endif

    GstPad *sinkpad, *srcpad;
    gchar pad_name_sink[16] = "sink_0";
    gchar pad_name_src[16] = "src";

    sinkpad = gst_element_get_request_pad (streammux, pad_name_sink);
    if (!sinkpad) {
        g_printerr ("Streammux request sink pad failed. Exiting.\n");
        return -1;
    }

    srcpad = gst_element_get_static_pad (decoder, pad_name_src);
    if (!srcpad) {
        g_printerr ("Decoder request src pad failed. Exiting.\n");
        return -1;
    }

    if (gst_pad_link (srcpad, sinkpad) != GST_PAD_LINK_OK) {
        g_printerr ("Failed to link decoder to stream muxer. Exiting.\n");
        return -1;
    }

    gst_object_unref (sinkpad);
    gst_object_unref (srcpad);

    /* we link the elements together */
    /* file-source -> h264-parser -> nvh264-decoder ->
     * nvinfer -> nvvidconv -> nvosd -> video-renderer */

    if (!gst_element_link_many (source, h264parser, decoder, NULL)) {
        g_printerr ("Elements could not be linked: 1. Exiting.\n");
        return -1;
    }

#ifdef PLATFORM_TEGRA
    if (!gst_element_link_many (streammux, pgie,
      nvvidconv, nvosd, transform, sink, NULL)) {
    g_printerr ("Elements could not be linked: 2. Exiting.\n");
    return -1;
  }
#else
    if (!gst_element_link_many (streammux, pgie, queue, sgie1, queue5, nvtracker,
                                nvvidconv, nvosd, sink, NULL)) {
        g_printerr ("Elements could not be linked: 2. Exiting.\n");
        return -1;
    }
#endif

    /* Lets add probe to get informed of the meta data generated, we add probe to
     * the sink pad of the osd element, since by that time, the buffer would have
     * had got all the metadata. */
    osd_sink_pad = gst_element_get_static_pad (nvosd, "sink");
    if (!osd_sink_pad)
        g_print ("Unable to get sink pad\n");
    else
        gst_pad_add_probe (osd_sink_pad, GST_PAD_PROBE_TYPE_BUFFER,
                           osd_sink_pad_buffer_probe, NULL, NULL);

    queue_src_pad = gst_element_get_static_pad (queue, "src");
    gst_pad_add_probe (queue_src_pad, GST_PAD_PROBE_TYPE_BUFFER,
                       pgie_pad_buffer_probe, NULL, NULL);

    tiler_sink_pad = gst_element_get_static_pad (queue5, "sink");
    gst_pad_add_probe (tiler_sink_pad, GST_PAD_PROBE_TYPE_BUFFER,
                       sgie_pad_buffer_probe, NULL, NULL);

    /* Set the pipeline to "playing" state */
    g_print ("Now playing: %s\n", filePath);
    gst_element_set_state (pipeline, GST_STATE_PLAYING);

    /* Wait till pipeline encounters an error or EOS */
    g_print ("Running...\n");
    g_main_loop_run (loop);

    /* Out of the main loop, clean up nicely */
    g_print ("Returned, stopping playback\n");
    gst_element_set_state (pipeline, GST_STATE_NULL);
    g_print ("Deleting pipeline\n");
    gst_object_unref (GST_OBJECT (pipeline));
    g_source_remove (bus_watch_id);
    g_main_loop_unref (loop);
    return 0;
}