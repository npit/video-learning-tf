from defs_ import *
from utils_ import *
import pickle
import json

class Validation:
    required_input = []
    def __init__(self, settings, logits):
        if not settings.val:
            return
        # frame-level logits
        self.logits = logits
        # clip / item - level logits
        self.item_logits = None
        self.validation_logits_save_counter = 0
        self.validation_logits_save_interval = settings.val.logits_save_interval

        # items refer to the primary unit we operate one, i.e. videos or frames
        self.item_logits = np.zeros([0, settings.num_classes], np.float32)
        self.item_labels = np.zeros([0, settings.num_classes], np.float32)
        # clips refers to image groups that compose a video, for training with clip information
        self.clip_logits = np.zeros([0, settings.num_classes], np.float32)
        self.clip_labels = np.zeros([0, settings.num_classes], np.float32)

        self.labels = tf.placeholder(tf.int32, [None, settings.num_classes], name="input_labels")
        self.required_input.append((self.labels, defs.net_input.labels, defs.dataset_tag.main))
        self.run_folder = settings.run_folder
        self.run_id = settings.run_id
        self.timestamp = settings.timestamp

    def process_description_validation_logits(self, logits, labels, dataset, fdict, padding):
        error("Not implemented")
        caption_lengths = fdict[self.caption_lengths]
        assert (len(logits) - padding == len(caption_lengths)), "Logits, labels length mismatch (%d, %d)" % (len(logits)-padding, len(caption_lengths))
        eos_index = dataset.vocabulary.index("EOS")
        # logits is words
        for idx in range(len(logits) - padding):
            image_logits = logits[idx,:]
            image_logits = image_logits[:dataset.max_caption_length]
            eos_position_binary_idx = [1 if x == eos_index else 0 for x in image_logits]

            if any(eos_position_binary_idx):
                # keep up to but not including eos. Get first index, if multiple .
                first_eos = eos_position_binary_idx.index(1)
                image_logits = image_logits[0:first_eos]
            # else, no EOS exists in the predicted caption
            # append the vector
            self.item_logits.append(image_logits)

        # get the labels. In validation mode, labels are EOS-free.
        cumulative_offset = 0
        for item_idx, cap_len in enumerate(caption_lengths):
            label_idxs = [ x + cumulative_offset for x in list(range(cap_len))]
            item_labels = labels[label_idxs,:]
            self.item_labels.append(item_labels)
            cumulative_offset = cumulative_offset + cap_len

    # validation accuracy computation
    def process_validation_logits(self, tag, settings, logits, fdict, padding):
        labels = fdict[self.labels]
        dataset = settings.feeder.get_dataset_by_tag(tag)[0]

        # batch item contains logits that correspond to whole clips. Accumulate to clip storage, and check for aggregation.
        if dataset.batch_item == defs.batch_item.clip:
            # per-clip logits in input : append to clip logits accumulator
            self.clip_logits = np.vstack((self.clip_logits, logits))
            self.clip_labels = np.vstack((self.clip_labels, labels))
            debug("Adding %d,%d clip logits and labels to a total of %d,%d." % (
                logits.shape[0], labels.shape[0], self.clip_logits.shape[0], self.clip_labels.shape[0]))

            cpv = dataset.clips_per_video[dataset.video_index]
            # while possible, pop a chunk for the current cpv, aggregate, and add to video logits accumulator
            while dataset.video_index < len(dataset.clips_per_video) and cpv <= len(self.clip_logits):

                # aggregate the logits and add to video logits accumulation
                self.apply_clip_fusion(self.clip_logits, cpv, self.clip_labels)
                # delete them from the accumulation
                self.clip_logits = self.clip_logits[cpv:,:]
                self.clip_labels = self.clip_labels[cpv:,:]

                debug("Aggregated %d clips to the %d-th video. Video accumulation is now %d,%d - clip accumulation is %d, %d." %
                      (dataset.clips_per_video[dataset.video_index], 1 + dataset.video_index, len(self.item_logits),
                       len(self.item_labels), len(self.clip_logits), len(self.clip_labels)))
                # advance video index
                dataset.video_index = dataset.video_index + 1
                if dataset.video_index >= len(dataset.clips_per_video):
                    break
                cpv = dataset.clips_per_video[dataset.video_index]
        else:
            # batch items are whole items of data
            if dataset.input_mode == defs.input_mode.video:
                # can directly fuse and append to video accumulators
                maxvid = dataset.batch_index * dataset.batch_size
                minvid = maxvid - dataset.batch_size

                for vidx in range(minvid, maxvid):
                    if vidx >= dataset.num_items:
                        break
                    cpv = dataset.clips_per_video[vidx]
                    debug("Aggregating %d clips for video %d in video batch mode" % (cpv, vidx + 1))
                    self.apply_clip_fusion(logits, cpv, labels, settings.val.clip_fusion_method)
                    logits = logits[cpv:,:]
                    labels = labels[cpv:,:]
                if not (len(logits) == 0 and len(labels) == 0):
                    error("Logits and/or labels non empty at the end of video item mode aggregation!")
                debug("Video logits and labels accumulation is now %d,%d video in video batch mode." %
                      (len(self.item_logits), len(self.item_labels)))
            else:
                # frames, simply append
                self.add_item_logits_labels(logits,labels)

    def save_validation_logits_chunk(self, save_all = False):
        # if saving is not enabled or no logits are stored, leave
        if self.validation_logits_save_interval is None or len(self.item_logits) == 0:
            return
        # if logits saving is set to once at the end
        if self.validation_logits_save_interval <= 0:
            # if we are at the end, save
            if save_all:
                # all extracted logits are in the container
                save_file = os.path.join(self.run_folder,"validation_logits_%s_%s.total" % (self.run_id, self.timestamp))
                info("Saving all %d extracted validation logits to %s" % (len(self.item_logits), save_file))
                with open(save_file, "wb") as ff:
                    pickle.dump(self.item_logits, ff)
            # else, just return
            return

        # if logits saving is done in batches, save either if batch is full or if it is the last step
        if len(self.item_logits) >= self.validation_logits_save_interval or save_all:
            save_file = os.path.join(self.run_folder,"validation_logits_%s_%s.part_%d" %
                                     ( self.run_id, self.timestamp, self.validation_logits_save_counter))
            info("Saving a %d-sized chunk of validation logits to %s" % (len(self.item_logits), save_file))
            with open(save_file, "wb") as f:
                pickle.dump(self.item_logits, f)
            # reset the container
            if type(self.item_logits) == np.ndarray:
                num_classes = int(self.item_logits.shape[-1])
                del self.item_logits
                self.item_logits = np.zeros([0, num_classes], np.float32)
            else:
                # list
                del self.item_logits
                self.item_logits = []

            self.validation_logits_save_counter += 1

    def load_validation_logits_chunk(self, chunk_idx):
        if self.validation_logits_save_interval is None:
            return self.item_logits
        save_file = os.path.join(self.run_folder,"validation_logits_%s_%s.part_%d" % ( self.run_id, self.timestamp, chunk_idx))
        with open(save_file, "rb") as f:
            logits_chunk = pickle.load(f)
        return logits_chunk

    def apply_clip_fusion(self, clips_logits, cpv, video_labels, clip_fusion):
        curr_clips = clips_logits[0:cpv,:]
        video_label = video_labels[0,:]
        if clip_fusion == defs.fusion_method.avg:
            video_logits = np.mean(curr_clips, axis=0)
        elif clip_fusion == defs.fusion_method.last:
            video_logits = curr_clips[-1, :]
            #elif clip_fusion == defs.fusion_method.
        # add logits, label to the video accumulation
        self.add_item_logits_labels(video_logits, video_label)

    def add_item_logits_labels(self,logits,label):
        # add logits, label to the video accumulation
        self.item_logits = np.vstack((self.item_logits, logits))
        self.item_labels = np.vstack((self.item_labels, label))

    def get_accuracy(self):
        # compute accuracy
        info("Computing accuracy")
        accuracies = []
        curr_item_idx = 0
        # compute partial accuracies for each saved chunk
        for saved_idx in range(self.validation_logits_save_counter):
            logits = self.load_validation_logits_chunk(saved_idx)
            chunk_size = len(logits)
            labels = self.item_labels[curr_item_idx:curr_item_idx + chunk_size, :]
            accuracies.append(self.get_chunk_accuracy(logits, labels))
            curr_item_idx += chunk_size
            info("Processed saved chunk %d/%d containing %d items - item total: %d" %
                 (saved_idx+1, self.validation_logits_save_counter, chunk_size, curr_item_idx))

        # compute partial accuracies for the unsaved chunk in item_logits
        if len(self.item_logits) > 0:
            chunk_size = len(self.item_logits)
            labels = self.item_labels[curr_item_idx:curr_item_idx + chunk_size, :]
            accuracies.append(self.get_chunk_accuracy(self.item_logits, labels))
            curr_item_idx += chunk_size
            info("Processed existing chunk containing %d items - item total: %d" % ( chunk_size, curr_item_idx))

        accuracy = np.mean(accuracies)
        return accuracy

    def get_chunk_accuracy(self, logits, labels):
        predicted_classes = np.argmax(logits, axis=1)
        correct_classes = np.argmax(labels, axis=1)
        return  np.mean(np.equal(predicted_classes, correct_classes))

    def process_description(self, settings):
        # get description metric
        # do an ifthenelse on the evaluation type (eg coco)

        # default eval. should be sth like a json production
        if settings.eval_type == defs.eval_type.coco:
            # evaluate coco
            # format expected is as per  http://mscoco.org/dataset/#format
            # [{ "image_id" : int, "caption" : str, }]

            # get captions from logits, write them in the needed format,
            # pass them to the evaluation function
            ids_captions = []
            num_processed_logits = 0

            for idx in range(self.validation_logits_save_counter):
                logits_chunk = self.load_validation_logits_chunk(idx)
                ids_captions_chunk = settings.validation_logits_to_captions(logits_chunk, num_processed_logits)
                ids_captions.extend(ids_captions_chunk)
                num_processed_logits  += len(logits_chunk)
                info("Processed saved chunk %d/%d containing %d items - item total: %d" %
                     (idx+1, self.validation_logits_save_counter, len(logits_chunk), num_processed_logits))
            if len(self.item_logits) > 0:
                error("Should never get item logits last chunk in runtask!!")
                ids_captions_chunk = settings.validation_logits_to_captions(self.item_logits, num_processed_logits)
                ids_captions.extend(ids_captions_chunk)
                info("Processed existing chunk containing %d items - item total: %d" % (len(self.item_logits), len(ids_captions)))

            # check for erroneous duplicates
            dupl = [obj["image_id"] for obj in ids_captions]
            if duplicates(dupl):
                error("Duplicate image ids in coco validation: %s" % str(dupl))

            # write results
            results_file = os.path.join(settings.run_folder, "coco.results.json")
            info("Writing captioning results to %s" % results_file)
            with open(results_file , "w") as fp:
                json.dump(ids_captions, fp)

            # also, get captions from the read image paths - labels files
            # initialize with it the COCO object
            # ....
            info("Evaluating captioning using ground truth file %s" % str(settings.caption_ground_truth))
            command = '$(which python2) tools/python2_coco_eval/coco_eval.py %s %s' % (results_file, settings.caption_ground_truth)
            debug("evaluation command is [%s]" % command)
            os.system(command)

