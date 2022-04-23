from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import torch
import string
import math


def isEnglish(token):
    return token in set(string.ascii_lowercase)


class Model:
    def __init__(self, repo_path, use_gpu=False):
        # load processor and model from huggingface
        self.processor = Wav2Vec2Processor.from_pretrained(repo_path)
        self.model = Wav2Vec2ForCTC.from_pretrained(repo_path)
        self.use_gpu = use_gpu

        # use gpu if specified
        if use_gpu:
            self.model.cuda()

    def extract_features(self, audio):
        """
        Extract features from audio inputs using wav2vec2.
        Converts 16000 Hz audio array into 49 Hz input values.
        """

        input_values = self.processor(
            audio, sampling_rate=16000, return_tensors="pt"
        ).input_values
        return input_values

    def get_tokens(self, audio):
        """
        Get array of predicted tokens from input.
        Converts 16000 Hz audio array into (approximately) 49 Hz output values.
        """
        input_values = self.extract_features(audio)

        # if gpu is used, send input values to the gpu first
        if self.use_gpu:
            input_values = input_values.cuda()

        # feed input into model
        with torch.no_grad():
            logits = self.model(input_values).logits

        # get output with maximum likelihood
        predicted_ids = torch.argmax(logits, dim=-1)

        # convert id to token
        tokens = self.processor.tokenizer.convert_ids_to_tokens(
            torch.flatten(predicted_ids)
        )
        return tokens

    def predict(self, audio):
        """
        Get predicted output from input audio.
        """
        input_values = self.extract_features(audio)

        # if gpu is used, send input values to the gpu first
        if self.use_gpu:
            input_values = input_values.cuda()

        # feed input into model
        with torch.no_grad():
            logits = self.model(input_values).logits

        # get output with maximum likelihood
        predicted_ids = torch.argmax(logits, dim=-1)

        # convert id to string
        prediction_str = self.processor.batch_decode(predicted_ids)[0]
        return prediction_str


class LID_Model(Model):
    def __init__(self, repo_path, use_gpu=False):
        super().__init__(repo_path, use_gpu=use_gpu)

        # get list of special tokens (padding, word delimiter, etc.)
        exception_tokens = self.processor.tokenizer.all_special_tokens
        if self.processor.tokenizer.word_delimiter_token != None:
            exception_tokens.append(self.processor.tokenizer.word_delimiter_token)

        self.exception_tokens_set = set(exception_tokens)

    def segment(self, audio):
        """
        Splits audio array into segments of different languages.
        Output format is an array with entries (language, segment_end).
        """
        tokens = self.get_tokens(audio)

        segments_in_tokens_frequency = []
        last_token_index = None
        # initialize default segment to silence
        curr_segment_language = "silence"
        for i in range(len(tokens)):
            curr_token = tokens[i]

            if curr_token not in self.exception_tokens_set:
                # if curr_token is the first token seen, update the entire segment
                if last_token_index == None:
                    if isEnglish(curr_token):
                        curr_segment_language = "english"
                    else:
                        curr_segment_language = "mandarin"

                    last_token_index = i
                    continue

                # if
                if curr_segment_language != "english" and isEnglish(curr_token):
                    # compute segment end as the middle between the last token index and curr token index
                    segment_end = (i + last_token_index) / 2
                    segments_in_tokens_frequency.append(
                        (curr_segment_language, segment_end)
                    )
                    # update the segment to the new language
                    curr_segment_language = "english"

                elif curr_segment_language != "mandarin" and not isEnglish(curr_token):
                    # compute segment end as the middle between the last token index and curr token index
                    segment_end = (i + last_token_index) / 2
                    segments_in_tokens_frequency.append(
                        (curr_segment_language, segment_end)
                    )
                    # update the segment to the new language
                    curr_segment_language = "mandarin"

                # update the index of the last token seen
                last_token_index = i

        # append the last segment, which could be silence if no tokens are seen
        segments_in_tokens_frequency.append((curr_segment_language, len(tokens)))

        segments = []
        # convert the segments splitting tokens to segments splitting audio
        for segment in segments_in_tokens_frequency:
            segment_language = segment[0]
            segment_end = math.floor((segment[1] / len(tokens)) * len(audio))
            segments.append((segment_language, segment_end))

        return segments


class _Combined_Model:
    # def __init__(
    #     self, lid_model: LID_Model, english_model: Model, mandarin_model: Model
    # ):
    #     self.lid_model = lid_model
    #     self.english_model = english_model
    #     self.mandarin_model = mandarin_model
	_instance = None

	def predict(self, audio):
		segments = self.lid_model.segment(audio)

		curr_segment_start = 0
		prediction = ""
		for segment in segments:
			curr_segment_language = segment[0]
			curr_segment_end = segment[1]
			if curr_segment_language == "english":
				prediction += self.english_model.predict(
					audio[curr_segment_start:curr_segment_end]
				)
			elif curr_segment_language == "mandarin":
				prediction += self.mandarin_model.predict(
					audio[curr_segment_start:curr_segment_end]
				)

			curr_segment_start = curr_segment_end

		return prediction


def Combined_Model():
# ensure an instance is created only the first time the factory function is called
	if _Combined_Model._instance is None:
		lid_model = LID_Model("ntoldalagi/nick_asr_v2")
		english_model = Model("jonatasgrosman/wav2vec2-large-xlsr-53-english")
		mandarin_model = Model("jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn")
		_Combined_Model._instance = _Combined_Model()
		_Combined_Model.lid_model = lid_model
		_Combined_Model.english_model = english_model
		_Combined_Model.mandarin_model = mandarin_model
		
		print("LOADING MODEL")
	return _Combined_Model._instance
	