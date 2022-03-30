import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import soundfile as sf
from scipy.io.wavfile import read
import numpy as np

class _Keyword_Spotting_Service:
    """Singleton class for keyword spotting inference with trained models.
    :param model: Trained model
    """

    model = None
    tokenizer = None
    _instance = None


    def predict(self, file_path, lastBlobStamp):
        """
        :param file_path (str): Path to audio file to predict
        :return predicted_keyword (str): Keyword predicted by the model
        """
        # print(file_path, os.path.isfile(file_path))
        # with contextlib.closing(wave.open(file_path,'r')) as f:
        #     frames = f.getnframes()
        #     rate = f.getframerate()
        #     duration = frames / float(rate)
        #     print(duration)
        
        audio_bytes = read(file_path)[1][int(lastBlobStamp * 2.47):]
        print("look here", lastBlobStamp, len(audio_bytes))
        audio_bytes = np.array(audio_bytes)
        x = torch.FloatTensor(audio_bytes)
        print(self.tokenizer)
        input_values = self.tokenizer(x, sampling_rate=16000, return_tensors='pt', padding='longest').input_values
        logits = self.model(input_values).logits
        tokens = torch.argmax(logits, axis=-1)
        texts = self.tokenizer.batch_decode(tokens)
        print(texts)
        return texts, len(audio_bytes)

    def preprocess(self, file_path, num_mfcc=13, n_fft=2048, hop_length=512):
        """Extract MFCCs from audio file.
        :param file_path (str): Path of audio file
        :param num_mfcc (int): # of coefficients to extract
        :param n_fft (int): Interval we consider to apply STFT. Measured in # of samples
        :param hop_length (int): Sliding window for STFT. Measured in # of samples
        :return MFCCs (ndarray): 2-dim array with MFCC data of shape (# time steps, # coefficients)
        """

        # load audio file
        return file_path


def Keyword_Spotting_Service():
    """Factory function for Keyword_Spotting_Service class.
    :return _Keyword_Spotting_Service._instance (_Keyword_Spotting_Service):
    """

    # ensure an instance is created only the first time the factory function is called
    if _Keyword_Spotting_Service._instance is None:
        _Keyword_Spotting_Service._instance = _Keyword_Spotting_Service()
        _Keyword_Spotting_Service.model = Wav2Vec2ForCTC.from_pretrained('jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn')
        _Keyword_Spotting_Service.tokenizer = Wav2Vec2Processor.from_pretrained('jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn')
        # _Keyword_Spotting_Service.model = Wav2Vec2ForCTC.from_pretrained('facebook/wav2vec2-base-960h')
        # _Keyword_Spotting_Service.tokenizer = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base-960h')
        # _Keyword_Spotting_Service.model = Wav2Vec2ForCTC.from_pretrained('facebook/wav2vec2-large-960h-lv60-self')
        # _Keyword_Spotting_Service.tokenizer = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-large-960h-lv60-self')
        print("LOADING MODEL")
    return _Keyword_Spotting_Service._instance


if __name__ == "__main__":

    # create 2 instances of the keyword spotting service
    kss = Keyword_Spotting_Service()
    kss1 = Keyword_Spotting_Service()

    # check that different instances of the keyword spotting service point back to the same object (singleton)
    assert kss is kss1

    # make a prediction
    keyword = kss.predict("audios/1646066594.5817962.wav")
    print(keyword)







# import librosa
# import tensorflow as tf
# import numpy as np

# SAVED_MODEL_PATH = "model.h5"
# SAMPLES_TO_CONSIDER = 22050

# class _Keyword_Spotting_Service:
#     """Singleton class for keyword spotting inference with trained models.
#     :param model: Trained model
#     """

#     model = None
#     _mapping = [
#         "down",
#         "off",
#         "on",
#         "no",
#         "yes",
#         "stop",
#         "up",
#         "right",
#         "left",
#         "go"
#     ]
#     _instance = None


#     def predict(self, file_path):
#         """
#         :param file_path (str): Path to audio file to predict
#         :return predicted_keyword (str): Keyword predicted by the model
#         """

#         # extract MFCC
#         MFCCs = self.preprocess(file_path)

#         # we need a 4-dim array to feed to the model for prediction: (# samples, # time steps, # coefficients, 1)
#         MFCCs = MFCCs[np.newaxis, ..., np.newaxis]

#         # get the predicted label
#         predictions = self.model.predict(MFCCs)
#         predicted_index = np.argmax(predictions)
#         predicted_keyword = self._mapping[predicted_index]
#         return predicted_keyword


#     def preprocess(self, file_path, num_mfcc=13, n_fft=2048, hop_length=512):
#         """Extract MFCCs from audio file.
#         :param file_path (str): Path of audio file
#         :param num_mfcc (int): # of coefficients to extract
#         :param n_fft (int): Interval we consider to apply STFT. Measured in # of samples
#         :param hop_length (int): Sliding window for STFT. Measured in # of samples
#         :return MFCCs (ndarray): 2-dim array with MFCC data of shape (# time steps, # coefficients)
#         """

#         # load audio file
#         signal, sample_rate = librosa.load(file_path)

#         if len(signal) >= SAMPLES_TO_CONSIDER:
#             # ensure consistency of the length of the signal
#             signal = signal[:SAMPLES_TO_CONSIDER]

#             # extract MFCCs
#             MFCCs = librosa.feature.mfcc(signal, sample_rate, n_mfcc=num_mfcc, n_fft=n_fft,
#                                          hop_length=hop_length)
#         return MFCCs.T


# def Keyword_Spotting_Service():
#     """Factory function for Keyword_Spotting_Service class.
#     :return _Keyword_Spotting_Service._instance (_Keyword_Spotting_Service):
#     """

#     # ensure an instance is created only the first time the factory function is called
#     if _Keyword_Spotting_Service._instance is None:
#         _Keyword_Spotting_Service._instance = _Keyword_Spotting_Service()
#         _Keyword_Spotting_Service.model = tf.keras.models.load_model(SAVED_MODEL_PATH)
#     return _Keyword_Spotting_Service._instance


# if __name__ == "__main__":

#     # create 2 instances of the keyword spotting service
#     kss = Keyword_Spotting_Service()
#     kss1 = Keyword_Spotting_Service()

#     # check that different instances of the keyword spotting service point back to the same object (singleton)
#     assert kss is kss1

#     # make a prediction
#     keyword = kss.predict("down.wav")
#     print(keyword)