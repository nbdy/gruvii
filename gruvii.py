from sys import argv
from os.path import isdir, exists
from os import listdir, makedirs, system
from pipes import quote
import numpy as np
import scipy.io.wavfile as wav
import tensorflow as tf


class Configuration(object):
    dataset_directory = None
    model_iterations = None
    sampling_frequency = None
    clip_length = None
    hidden_dimensions = None
    epochs = None

    def __init__(self):
        self.dataset_directory = "./dataset/test/"
        self.model_iterations = 50
        self.sampling_frequency = 44100
        self.clip_length = 10
        self.hidden_dimensions = 1024
        self.batch_size = 5
        self.epochs = 25

    @staticmethod
    def help():
        print("usage: gruvii.py {arguments}")
        print("{arguments}\t\t{default value}")
        print("\t--help")
        print("\t-d\t--dataset-directory\t./dataset/test/")
        print("\t-i\t--iterations\t50")
        print("\t-s\t--sampling-frequency\t44100")
        print("\t-c\t--clip-length\t10")
        print("\t-h\t--hidden-dimensions\t1024")
        print("\t-b\t--batch-size\t5")
        print("\t-e\t--epochs\t25")
        exit()

    @staticmethod
    def parse():
        c = Configuration()
        i = 0
        while i < len(argv):
            a = argv[i]
            if a in ["--help"]:
                Configuration.help()
            elif a in ["-d", "--dataset-directory"]:
                c.dataset_directory = argv[i + 1]
            elif a in ["-i", "--iterations"]:
                c.model_iterations = int(argv[i + 1])
            elif a in ["-s", "--sampling-frequency"]:
                c.sampling_frequency = int(argv[i + 1])
            elif a in ["-c", "--clip-length"]:
                c.clip_length = int(argv[i + 1])
            elif a in ["-h", "--hidden-dimensions"]:
                c.hidden_dimensions = int(argv[i + 1])
            elif a in ["-b", "--batch-size"]:
                c.batch_size = int(argv[i + 1])
            elif a in ["-e", "--epochs"]:
                c.epochs = int(argv[i + 1])
            i += 1
        return c


class Trainer(object):
    config = None

    block_size = None
    max_seq_length = None

    def __init__(self, config):
        self.config = config
        self._calc()

    def _calc(self):
        self.block_size = self.config.sampling_frequency / 4
        self.max_seq_length = int(round((self.config.sampling_frequency * self.config.clip_length) / self.block_size))

    def prepare_data(self):
        print("preparing data")
        nd = self.convert_folder_to_wav(self.config.dataset_directory, self.config.sampling_frequency)
        print("wrote waves to", nd)
        if self.config.dataset_directory.endswith("/"):
            of = self.config.dataset_directory.split("/")[-2]
        else:
            of = self.config.dataset_directory.split("/")[-1]
        print("output file prefix:", of)
        self.convert_wav_files_to_nptensor(nd, self.block_size, self.max_seq_length, of)
        return of

    @staticmethod
    def convert_folder_to_wav(directory, sample_rate=44100):
        od = directory + "wave/"
        if isdir(od):
            return od
        for file in listdir(directory):
            full_filename = directory + file
            if file.endswith('.mp3'):
                Trainer.convert_mp3_to_wav(filename=full_filename, sample_frequency=sample_rate)
            if file.endswith('.flac'):
                Trainer.convert_flac_to_wav(filename=full_filename, sample_frequency=sample_rate)
        return od

    @staticmethod
    def convert_flac_to_wav(filename, sample_frequency):
        new_path, tmp_path, orig_filename = Trainer.filter_ext(".flac", filename)
        new_path += 'wave'
        if not exists(new_path):
            makedirs(new_path)
        new_name = new_path + '/' + orig_filename + '.wav'
        cmd = 'sox {0} {1} channels 1 rate {2}'.format(quote(filename), quote(new_name), sample_frequency)
        system(cmd)
        return new_name

    @staticmethod
    def filter_ext(ext, filename):
        ext = filename[-len(ext):]
        if ext != ext:
            return
        files = filename.split('/')
        orig_filename = files[-1][0:-len(ext)]
        new_path = ''
        if filename[0] == '/':
            new_path = '/'
        for i in range(len(files) - 1):
            new_path += files[i] + '/'
        tmp_path = new_path + 'tmp'
        new_path += 'wave'
        return new_path, tmp_path, orig_filename

    @staticmethod
    def convert_mp3_to_wav(filename, sample_frequency):
        new_path, tmp_path, orig_filename = Trainer.filter_ext(".mp3", filename)
        if not exists(new_path):
            makedirs(new_path)
        if not exists(tmp_path):
            makedirs(tmp_path)
        filename_tmp = tmp_path + '/' + orig_filename + '.mp3'
        new_name = new_path + '/' + orig_filename + '.wav'
        sample_freq_str = "{0:.1f}".format(float(sample_frequency) / 1000.0)
        cmd = 'lame -a -m m {0} {1}'.format(quote(filename), quote(filename_tmp))
        system(cmd)
        cmd = 'lame --decode {0} {1} --resample {2}'.format(quote(filename_tmp), quote(new_name), sample_freq_str)
        system(cmd)
        return new_name

    @staticmethod
    def read_wav_as_np(filename):
        data = wav.read(filename)
        np_arr = data[1].astype('float32') / 32767.0  # Normalize 16-bit input to [-1, 1] range
        np_arr = np.array(np_arr)
        return np_arr, data[0]

    @staticmethod
    def convert_np_audio_to_sample_blocks(song_np, block_size):
        song_np = song_np.astype('int')
        block_lists = []
        total_samples = song_np.shape[0]
        num_samples_so_far = 0
        while num_samples_so_far < total_samples:
            block = song_np[num_samples_so_far:num_samples_so_far + int(block_size)]
            if block.shape[0] < block_size:
                padding = np.zeros((int(block_size) - block.shape[0]))
                block = np.concatenate((block, padding))
            block_lists.append(block)
            num_samples_so_far += block_size
            num_samples_so_far = int(num_samples_so_far)
        return block_lists

    @staticmethod
    def time_blocks_to_fft_blocks(blocks_time_domain):
        fft_blocks = []
        for block in blocks_time_domain:
            fft_block = np.fft.fft(block)
            new_block = np.concatenate((np.real(fft_block), np.imag(fft_block)))
            fft_blocks.append(new_block)
        return fft_blocks

    @staticmethod
    def load_training_example(filename, block_size=2048, use_time_domain=False):
        data, bitrate = Trainer.read_wav_as_np(filename)
        x_t = Trainer.convert_np_audio_to_sample_blocks(data, block_size)
        y_t = x_t[1:]
        y_t.append(np.zeros(int(block_size)))  # Add special end block composed of all zeros
        if use_time_domain:
            return x_t, y_t
        x = Trainer.time_blocks_to_fft_blocks(x_t)
        y = Trainer.time_blocks_to_fft_blocks(y_t)
        return x, y

    @staticmethod
    def convert_wav_files_to_nptensor(directory, block_size, max_seq_len, out_file, max_files=20,
                                      use_time_domain=False):
        files = []
        for file in listdir(directory):
            if file.endswith('.wav'):
                files.append(directory + file)
        print("converting", files, "to nptensors")
        chunks_x = []
        chunks_y = []
        num_files = len(files)
        if num_files > max_files:
            num_files = max_files
        for file_idx in range(num_files):
            file = files[file_idx]
            print('Processing: ', (file_idx + 1), '/', num_files)
            print('Filename: ', file)
            x, y = Trainer.load_training_example(file, block_size, use_time_domain=use_time_domain)
            cur_seq = 0
            total_seq = len(x)
            print("total_seq:", total_seq, "max_seq_len:", max_seq_len)
            while cur_seq + max_seq_len < total_seq:
                chunks_x.append(x[cur_seq:cur_seq + max_seq_len])
                chunks_y.append(y[cur_seq:cur_seq + max_seq_len])
                cur_seq += max_seq_len
        num_examples = len(chunks_x)
        num_dims_out = block_size * 2
        if use_time_domain:
            num_dims_out = block_size
        out_shape = (num_examples, max_seq_len, int(num_dims_out))
        x_data = np.zeros(out_shape, "i")
        y_data = np.zeros(out_shape, "i")
        for n in range(num_examples):
            for i in range(max_seq_len):
                x_data[n][i] = chunks_x[n][i]
                y_data[n][i] = chunks_y[n][i]
            print('Saved example ', (n + 1), ' / ', num_examples)
        print('Flushing to disk...')
        mean_x = np.mean(np.mean(x_data, axis=0), axis=0)  # Mean across num examples and num timesteps
        std_x = np.sqrt(np.mean(np.mean(np.abs(x_data - mean_x) ** 2, axis=0), axis=0))
        std_x = np.maximum(1.0e-8, std_x)  # Clamp variance if too tiny
        x_data[:][:] = (x_data[:][:] - mean_x)  # Mean 0
        x_data[:][:] = (x_data[:][:] / std_x)  # Variance 1
        y_data[:][:] = (y_data[:][:] - mean_x)  # Mean 0
        y_data[:][:] = (y_data[:][:] / std_x)  # Variance 1

        np.save(out_file + '_mean', mean_x)
        np.save(out_file + '_var', std_x)
        np.save(out_file + '_x', x_data)
        np.save(out_file + '_y', y_data)
        print('Done!')

    def train(self, prefix):
        print("loading training data")
        x_t = np.load(prefix + "_x.npy")
        y_t = np.load(prefix + "_y.npy")
        print("loaded training data")
        frq_space_dims = x_t.shape[2]
        print("got", frq_space_dims, "frequency dimensions")
        print("building model")
        model = tf.keras.models.Sequential([
            tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(frq_space_dims)),
            tf.keras.layers.LSTM(self.config.hidden_dimensions, return_sequences=True),
            tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(frq_space_dims))
        ])
        print("compiling model")
        model.compile(loss="mean_squared_error", optimizer="rmsprop")
        i = 0
        while i < self.config.model_iterations:
            print("iteration:", i)
            model.fit(x_t, y_t, self.config.batch_size, self.config.epochs)
            i += self.config.epochs
        model.save_weights(prefix + str(i))


if __name__ == '__main__':
    cfg = Configuration.parse()
    print("config:", cfg.__dict__)
    t = Trainer(cfg)
    npy_prefix = t.prepare_data()
    t.train(npy_prefix)
