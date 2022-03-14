import os
import wave
import contextlib
import tensorflow as tf

SAVED_MODEL_PATH = "/Users/Chen/.cache/huggingface/hub/a859443a82cbdaec31886961566f75b08a4a44ead30ce1608b506f0582b18902.274e30cb59cb39875f0e4f14d02de67cd3491470716f1be9205ab9f86265448d.h5"

class _Model_runner:
    """Singleton class for managing multiple model runner instances
    :param model: Trained model
    """

    model = None
    busy = False

    def predict(self, file_path):
        """
        :param file_path (str): Path to audio file to predict
        :return predicted_keyword (str): Keyword predicted by the model
        """
        self.busy = True
        res = self.model.predict(file_path)
        self.busy = False

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

class _Model_manager:
    """Singleton class for managing multiple model runner instances
    :param model: Trained model
    """

    model = None
    _instance = None
    runner_instances = []

    # def spawn(self, n):
    #     for i in range(n):
    #         self.runner_instances.append(_Model_runner())
    


def Keyword_Spotting_Service():
    """Factory function for Keyword_Spotting_Service class.
    :return _Keyword_Spotting_Service._instance (_Keyword_Spotting_Service):
    """

    # ensure an instance is created only the first time the factory function is called
    if _Keyword_Spotting_Service._instance is None:
        _Keyword_Spotting_Service._instance = _Keyword_Spotting_Service()
        _Keyword_Spotting_Service.model = tf.keras.models.load_model(SAVED_MODEL_PATH)
        print("model print", _Keyword_Spotting_Service.model)
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