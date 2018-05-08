from defs_ import *
import tensorflow as tf
import pickle
from utils_ import  *
from tools.inspect_checkpoint import get_checkpoint_tensor_names
import dataset_

class Feeder:
    """
    Class to handle save/load and data feeding to the models

    Input and delegation from here.
    feeddict from here, too.
    """
    def __init__(self, workflow, input_mode, phases, trainval, save_freq_per_epoch, run_folder, resume):
        self.inputs = []
        self.datasets = {}

        self.input_mode = input_mode
        self.workflow = workflow
        self.phases = phases
        self.phase = None

        self.run_folder = run_folder
        self.resume = resume

        self.train, self.val = trainval
        self.save_freq_per_epoch = save_freq_per_epoch


    def add_dataset(self, dataset_phase, id, path, mean_image, prepend_folder, image_shape, imgproc, raw_image_shape, data_format,
                    frame_format, batch_item, num_classes, tag, read_tries, captioning_config = None):
        dset = dataset_.Dataset()
        if not dataset_phase in self.datasets:
            self.datasets[dataset_phase] = []
        self.datasets[dataset_phase].append(dset)
        dset.initialize(id, path, mean_image, prepend_folder, image_shape, imgproc, raw_image_shape, data_format,
                            frame_format, batch_item, num_classes, tag, read_tries)
        # embeddings file
        if captioning_config:
            word_embeddings_file = captioning_config[-1]
            dset.initialize_workflow(word_embeddings_file[-1])


    def set_phase(self, phase):
        self.phase = phase

    def initialize_datasets(self):
        #if not self.phase:
        #    error("No phase set to initialize datasets to.")
        if not self.datasets:
            error("No dataset configured to active phase [%s]" % self.phase)
        for phase in self.phases:
            for i, dset in enumerate(self.datasets[phase]):
                info("Reading dataset %d / %d : [%s]" % (i+1, len(self.datasets[phase]), dset.id))
                if defs.phase.train in self.phases and self.train:
                    dset.calculate_batches(self.train.batch_size, self.input_mode)
                elif defs.phase.val in self.phases and self.val:
                    dset.calculate_batches(self.val.batch_size, self.input_mode)

    def loop(self):
        # it is assumed that all datasets are loop-synchronized
        return self.datasets[self.phase][0].loop()

    def get_dataset_by_tag(self, tag):
        dsets = [dset for dset in self.datasets[self.phase] if dset.tag == tag]
        return dsets

    def get_num_batches(self):
        if not self.datasets:
            return -1
        return len(self.datasets[self.phase][0].batches)


    def validation_logits_to_captions(self, logits_chunk, num_processed_logits):
        return self.datasets[self.phase].validation_logits_to_captions(logits_chunk, num_processed_logits)


    def get_next_batch(self):
        images, ground_truth, ids = [],[],[]
        for dset in self.datasets[self.phase]:
            im, g = dset.get_next_batch()
            images.append(im)
            ground_truth.append(g)
            ids.append(dset.id)
        return images, ground_truth, ids


    def get_feed_dict(self, required_input):

        images, ground_truth, dataset_ids = self.get_next_batch()
        fdict = {}
        num_labels = None
        num_data = [len(im) for im in images]
        # get the required input(s) from the batch
        for req_input in required_input:
            i_tens, i_type, i_datatag = req_input
            dataset_id = [ dset.id for dset in self.datasets[self.phase] if dset.tag == i_datatag]
            if not (len(dataset_id) == 1):
                error("%d datasets satisfy the following network input requirement, but exactly one must. %s." % (len(dataset_id), str(required_input)))
            dataset_idx = dataset_ids.index(dataset_id[0])
            if i_type == defs.net_input.visual:
                fdict[i_tens] = images[dataset_idx]
            elif i_type == defs.net_input.labels:
                fdict[i_tens] = ground_truth[dataset_idx]
                num_labels = len(ground_truth[dataset_idx])

        assert num_labels is not None, "Unset num. labels in feed dict!"
        padding = 0
        # for description workflows, supply wordvectors and caption lengths
        #if defs.workflows.is_description(settings.workflow):
            # get words per caption, onehot labels, embeddings
        #    fdict[model.inputLabels] = ground_truth["onehot_labels"]
        #    fdict[model.caption_lengths] = ground_truth['caption_lengths']
        #    fdict[model.word_embeddings] = ground_truth['word_embeddings']
        #    fdict[model.non_padding_word_idxs] = ground_truth['non_padding_index']
        #    num_labels = len(ground_truth["onehot_labels"])
        #else:
        #    fdict[lrcn.inputLabels] = ground_truth
        #    num_labels = len(ground_truth)

        return fdict, num_data, num_labels, padding




    def should_save(self, step):
        if self.save_interval < 0 or self.phase == defs.phase.val:
            return False
        return step % self.save_interval == 0


    def get_batch_sizes(self):
        batch_sizes = []
        for dset in self.datasets[self.phase]:
            batch_sizes.append(dset.batch_size)
        return  batch_sizes

    def compute_save_interval(self):
        if not self.train:
            self.save_interval, self.num_saves = -1, 0
            return
        # just check the first
        for dset in self.datasets[self.phase]:
            self.save_interval, self.num_saves = dset.compute_dataset_portion(self.save_freq_per_epoch, self.train.epochs)

    def get_batch_index(self):
        return self.datasets[self.phase][0].batch_index

    def rewind_datasets(self):
        for dset in self.datasets[self.phase]:
            dset.rewind()
    def get_datasets(self):
        return self.datasets[self.phase]



    # restore dataset meta parameters
    def resume_snap(self, resume_file):
        if not self.resume:
           return
        if resume_file == defs.names.latest_savefile:
            with open(os.path.join(self.run_folder,"checkpoints","checkpoint"),"r") as f:
                for line in f:
                    savefile_graph = line.strip().split(maxsplit=1)[-1].strip()
                    if savefile_graph[::len(savefile_graph)-1] == '""': savefile_graph = savefile_graph[1:-1]
                    savefile_metapars = savefile_graph + ".snap"
                    msg = "Resuming latest tf metadata: [%s]" % savefile_metapars
                    break
        else:
            savefile_metapars = resume_file + ".snap"
            msg = "Resuming specified tf metadata: [%s]" % savefile_metapars

        info(msg)
        if not os.path.exists(savefile_metapars):
            error("Metaparameters savefile does not exist: %s" %  savefile_metapars)
        try:
            # load saved parameters pickle
            with open(savefile_metapars, 'rb') as f:
                params = pickle.load(f)
        except Exception as ex:
            error(ex)

        # set run options from loaded stuff
        batch_info, epoch = params[:2]
        # assign global step
        try:
            global_step = params[2]
        except:
            # parse from filename
            global_step_str = os.path.basename(savefile_metapars).split(".")[-2].split("-")[-1]
            global_step = int(global_step_str)

        #if defs.workflows.is_description(self.workflow):
        #    self.sequence_length = params[2:]

        # inform datasets - if batch index info is paired with a dataset id, inform that dataset. Else, inform the 1st
        for dset in self.get_datasets():
            idx = 0
            if type(batch_info) == dict:
                if dset.tag in batch_info:
                    idx = batch_info[dset.tag]
            else:
                # an int - update it regardless
                idx = batch_info
            dset.restore(idx, epoch)

        info("Restored training snapshot of epoch %d, train index %s, global step %d" % (epoch+1, str(batch_info), global_step))
        return epoch, global_step


    # restore graph variables
    def init_saveload(self, sess, resume_file, ignorable_variable_names):
        # initialize graph saving / loading
        self.compute_save_interval()
        self.saver = tf.train.Saver(max_to_keep = self.num_saves)

        if self.phase == defs.phase.train and self.num_saves <= 0:
            return
        if self.resume:
            debug("Handling resume options")
            if resume_file == defs.names.latest_savefile:
                with open(os.path.join(self.run_folder,"checkpoints","checkpoint"),"r") as f:
                    for line in f:
                        savefile_graph = line.strip().split(maxsplit=1)[-1].strip()
                        msg = "Resuming latest tf graph: [%s]" % savefile_graph
                        break
            else:
                savefile_graph = os.path.join(resume_file)
                msg = "Resuming specified tf graph: [%s]" % savefile_graph

            # handle surrounding quotes
            if savefile_graph.startswith('"') or savefile_graph.startswith("'"): savefile_graph = savefile_graph[1:-1]
            info(msg)
            required_files = [savefile_graph + "." + suf for suf in ["meta","index","snap"]]
            exists = [os.path.exists(f) for f in required_files]
            if any([ not ex for ex in exists]) :
                for fname, ex in zip(required_files, exists):
                    print("file: [%s], exists: [%s]" % (fname, str(ex)))
                error("Missing meta, index or snap part: [%s], from graph savefile: %s" % (str(exists), savefile_graph))

            try:
                # if we are in validation mode, the 'global_step' training variable is discardable
                if self.val:
                    ignorable_variable_names.append(defs.names.global_step)

                chkpt_names = get_checkpoint_tensor_names(savefile_graph)
                # get all variables the project, omitting the :<num> appendices
                curr_names = [ drop_tensor_name_index(t.name) for t in tf.global_variables()]
                names_missing_from_chkpt = [n for n in curr_names if n not in chkpt_names and n not in ignorable_variable_names]
                names_missing_from_curr = [n for n in chkpt_names if n not in curr_names and n not in ignorable_variable_names]

                if names_missing_from_chkpt:
                    missing_unignorables = [n for n in names_missing_from_chkpt if not n in ignorable_variable_names]
                    warning("Found %d unignorable variables missing from checkpoint:[%s]" %
                            (len(missing_unignorables),missing_unignorables))
                    # Better warn the user and await input
                    ans = input("Continue? (y/n)")
                    if ans != "y":
                        error("Failed to load checkpoint")
                if names_missing_from_curr:
                    warning("There are checkpoint variables missing in the project:[%s]" % names_missing_from_curr)
                    # Better warn the user and await input
                    ans = input("Continue? (y/n)")
                    if ans != "y":
                        error("Failed to load checkpoint")
                # load saved graph file
                tf.reset_default_graph()
                self.saver.restore(sess, savefile_graph)
            except tf.errors.NotFoundError as err:
                # warning(err.message)
                pass
            except:
                error("Failed to load checkpoint!")

    def get_dataset_item_shape(self):
        return self.sh

    # save graph and dataset stuff
    def save(self, sess, progress, global_step):
        try:
            # save the graph
            checkpoints_folder = os.path.join(self.run_folder, "checkpoints")
            if not os.path.exists(checkpoints_folder):
                os.makedirs(checkpoints_folder)

            basename = os.path.join(checkpoints_folder, get_datetime_str() + "_" + self.workflow + "_" + progress)
            savefile_graph = basename + ".graph"

            info("Saving graph  to [%s]" % savefile_graph)
            saved_instance_name = self.saver.save(sess, savefile_graph, global_step=global_step)

            # save dataset metaparams
            savefile_metapars = saved_instance_name + ".snap"

            info("Saving params to [%s]" % savefile_metapars)
            info("Saving params for epoch index %d, train index %d" %
                 (self.train.epoch_index + 1, self.get_batch_index()))

            params2save = [self.get_batch_index(), self.train.epoch_index, global_step]
            if defs.workflows.is_description(self.workflow):
                params2save += [ dat.max_caption_length for dat in self.get_datasets()]

            with open(savefile_metapars,'wb') as f:
                pickle.dump(params2save,f)
        except Exception as ex:
            error(ex)
