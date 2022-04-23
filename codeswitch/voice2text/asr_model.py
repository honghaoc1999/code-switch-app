import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import soundfile as sf
from scipy.io.wavfile import read, write
import numpy as np
from pydub import AudioSegment
from pydub.silence import split_on_silence, detect_nonsilent
import uuid
from autocorrect import Speller
import re

class _Keyword_Spotting_Service:
    """Singleton class for keyword spotting inference with trained models.
    :param model: Trained model
    """

    model = None
    tokenizer = None
    _instance = None


    def preprocess(self, file_path, silentChunkNum, lastChunk, num_mfcc=13, n_fft=2048, hop_length=512):
        """Extract MFCCs from audio file.
        :param file_path (str): Path of audio file
        :param num_mfcc (int): # of coefficients to extract
        :param n_fft (int): Interval we consider to apply STFT. Measured in # of samples
        :param hop_length (int): Sliding window for STFT. Measured in # of samples
        :return MFCCs (ndarray): 2-dim array with MFCC data of shape (# time steps, # coefficients)
        """

        # Load your audio.
        song = AudioSegment.from_wav(file_path)
        def match_target_amplitude(aChunk, target_dBFS):
            ''' Normalize given audio chunk '''
            change_in_dBFS = target_dBFS - aChunk.dBFS
            return aChunk.apply_gain(change_in_dBFS)

        
        # Split track where the silence is 2 seconds or more and get chunks using 
        # the imported function.
        print("song.dBFS", song.dBFS)
        chunks = split_on_silence (
            # Use the loaded audio.
            song, 
            # Specify that a silent chunk must be at least 2 seconds or 2000 ms long.
            min_silence_len = 150,
            silence_thresh = song.dBFS - 16,
            keep_silence = True,
            seek_step=1
            # Consider a chunk silent if it's quieter than -16 dBFS.
            # (You may want to adjust this parameter.)
        )
        ranges = detect_nonsilent(
            song, 
            min_silence_len=150, 
            silence_thresh=song.dBFS-16, 
            seek_step=1
        )
        print("sanity check: ",len(chunks), (len(ranges)))
        rightMostTranscribeChunkIndex = None
        if lastChunk == 'true':
            rightMostTranscribeChunkBound = len(chunks)
            print("this is last chunk")
        else:
            print("silent chunks num: ", silentChunkNum, len(chunks))
            rightMostTranscribeChunkBound = len(chunks) - 1
        
        chunk_filepaths = []
        for i in range(silentChunkNum, rightMostTranscribeChunkBound):
            chunk_filepath = "voice2text/chunks/" + str(i) + '_' + str(uuid.uuid1()) + '.wav'
            chunks[i].export(out_f = chunk_filepath, 
                        format = "wav")
            chunk_filepaths.append(chunk_filepath)
        
        return chunk_filepaths, len(chunks) - 1
        

        # Process each chunk with your parameters
        # for i, chunk in enumerate(chunks):
        #     # Create a silence chunk that's 0.5 seconds (or 500 ms) long for padding.
        #     silence_chunk = AudioSegment.silent(duration=500)

        #     # Add the padding chunk to beginning and end of the entire chunk.
        #     audio_chunk = silence_chunk + chunk + silence_chunk

        #     # Normalize the entire chunk.
        #     normalized_chunk = match_target_amplitude(audio_chunk, -20.0)

        #     # Export the audio chunk with new bitrate.
        #     print("Exporting chunk{0}.mp3.".format(i))
        #     normalized_chunk.export(
        #         ".//chunk{0}.mp3".format(i),
        #         bitrate = "192k",
        #         format = "mp3"
        #     )
    def correct_eng_chn_mix_seg(self, s, speller):
        res = []
        last_bound = 0
        for i in range(len(s)):
            if i == 0:
                continue
            if '\u4e00' <= s[i] <= '\u9fef':
                if not '\u4e00' <= s[i-1] <= '\u9fef':
                    res.append(speller(s[last_bound:i]))
                    last_bound = i
            else:
                if '\u4e00' <= s[i-1] <= '\u9fef':
                    res.append(s[last_bound:i])
                    last_bound = i
        if '\u4e00' <= s[last_bound] <= '\u9fef':
            res.append(s[last_bound:])
        else:
            res.append(speller(s[last_bound:]))
        print("corrected mix: ", ' '.join(res))
        return ' '.join(res)



    def cleanup(self, text):
        text = re.sub("\嗯+", " um ", text[0])
        text = re.sub("\呃+", " um ", text)
        segments = text.split(' ')
        spell = Speller()
        for i in range(len(segments)):
            if len(re.findall(r'[\u4e00-\u9fff]+',segments[i])) == 0:
                segments[i] = spell(segments[i].lower())
            else:
                segments[i] = self.correct_eng_chn_mix_seg(segments[i], spell)
            
        return [' '.join(segments)]

    def predict(self, file_path, lastBlobStamp, runFull, lastChunk=None, silentChunkNum=None, runEng="true"):
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
        # ratio = self.preprocess(file_path)
        full_audio = read(file_path)[1]
        newSilentChunkNum = None
        if runFull == 'true':
            audio_bytes = full_audio
        else:
            if silentChunkNum != None:
                chunk_filepaths, newSilentChunkNum = self.preprocess(file_path, silentChunkNum, lastChunk)
                print("right after process: ", newSilentChunkNum)
                if len(chunk_filepaths) == 0:
                    return "", -1, newSilentChunkNum
                audio_bytes = []
                for filepath in chunk_filepaths:
                    print("check here", type(read(filepath)[1]), len(read(filepath)[1]))
                    if len(audio_bytes) == 0:
                        audio_bytes = read(filepath)[1]
                    else:
                        audio_bytes = np.concatenate((audio_bytes,read(filepath)[1]))

            else: 
                # audio_bytes = full_audio[int(lastBlobStamp * 2.51329556):]
                audio_bytes = full_audio[int(lastBlobStamp * 2.48):]

                write("voice2text/audio_chunks/"+file_path[len("voice2text/audios/"):], 16000, audio_bytes)
            # last_chunk_len = int(len(full_audio) * ratio)
            # audio_bytes = full_audio[:-last_chunk_len]
            # print("look here", len(audio_bytes),int(lastBlobStamp * 2.5132956))
        audio_bytes = np.array(audio_bytes)
        x = torch.FloatTensor(audio_bytes)
        if runEng == 'true' or runFull == 'true':
            
            input_values = self.tokenizer(x, sampling_rate=16000, return_tensors='pt', padding='longest').input_values
            logits = self.model(input_values).logits
            tokens = torch.argmax(logits, axis=-1)
            # print("audiolen vs. logits vs. tokens vs. input_values", len(audio_bytes), logits.shape, tokens.shape, input_values.shape)
            # print(tokens, self.tokenizer.convert_ids_to_tokens(list(tokens)))
            # if runFull == 'true':
            #     np.savetxt('model_output.txt', tokens.numpy())
            texts = self.tokenizer.batch_decode(tokens)
            texts = self.cleanup(texts)
            if len(re.findall(r'[\u4e00-\u9fff]+',texts[0])) == 0:
                eng_input_values = self.eng_tokenizer(x, sampling_rate=16000, return_tensors='pt', padding='longest').input_values
                eng_logits = self.eng_model(eng_input_values).logits
                eng_tokens = torch.argmax(eng_logits, axis=-1)
                eng_texts = self.eng_tokenizer.batch_decode(eng_tokens)
                eng_texts = self.cleanup(eng_texts)
                texts = eng_texts

        else:
            input_values = self.simple_tokenizer(x, sampling_rate=16000, return_tensors='pt', padding='longest').input_values
            logits = self.simple_model(input_values).logits
            tokens = torch.argmax(logits, axis=-1)
            texts = self.simple_tokenizer.batch_decode(tokens)
            texts = self.cleanup(texts)
        return texts, len(audio_bytes), newSilentChunkNum

    


def Keyword_Spotting_Service():
    """Factory function for Keyword_Spotting_Service class.
    :return _Keyword_Spotting_Service._instance (_Keyword_Spotting_Service):
    """
    # ensure an instance is created only the first time the factory function is called
    if _Keyword_Spotting_Service._instance is None:
        _Keyword_Spotting_Service._instance = _Keyword_Spotting_Service()
        # _Keyword_Spotting_Service.model = Wav2Vec2ForCTC.from_pretrained('jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn')
        # _Keyword_Spotting_Service.tokenizer = Wav2Vec2Processor.from_pretrained('jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn')
        _Keyword_Spotting_Service.simple_model = Wav2Vec2ForCTC.from_pretrained('facebook/wav2vec2-base-960h')
        _Keyword_Spotting_Service.simple_tokenizer = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base-960h')
        # _Keyword_Spotting_Service.model = Wav2Vec2ForCTC.from_pretrained('facebook/wav2vec2-large-960h-lv60-self')
        # _Keyword_Spotting_Service.tokenizer = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-large-960h-lv60-self')
        # _Keyword_Spotting_Service.model = Wav2Vec2ForCTC.from_pretrained('GleamEyeBeast/ascend')
        # _Keyword_Spotting_Service.tokenizer = Wav2Vec2Processor.from_pretrained('GleamEyeBeast/ascend')
        _Keyword_Spotting_Service.model = Wav2Vec2ForCTC.from_pretrained('ntoldalagi/nick_asr_v2')
        _Keyword_Spotting_Service.tokenizer = Wav2Vec2Processor.from_pretrained('ntoldalagi/nick_asr_v2')
        _Keyword_Spotting_Service.eng_model = Wav2Vec2ForCTC.from_pretrained('facebook/wav2vec2-large-960h-lv60-self')
        _Keyword_Spotting_Service.eng_tokenizer = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-large-960h-lv60-self')
        # _Keyword_Spotting_Service.lid_model = Wav2Vec2ForCTC.from_pretrained('ntoldalagi/nick_asr_LID')
        # _Keyword_Spotting_Service.lid_tokenizer = Wav2Vec2Processor.from_pretrained('ntoldalagi/nick_asr_LID')
        
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