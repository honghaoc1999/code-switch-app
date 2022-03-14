from django.shortcuts import render
import wave
import json
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import base64
import subprocess
import time
import voice2text.asr_model as asr_model
# Create your views here.
def recorderView(request):
    context = {}
    return render(request, 'recorder.html', context)

def convert_webm_to_wav(webmFile, wavFile):
    command = ['ffmpeg', '-i', webmFile, '-acodec', 'pcm_s16le', '-ac', '1', '-ar', '16000', wavFile]
    subprocess.run(command,stdout=subprocess.PIPE,stdin=subprocess.PIPE)


@csrf_exempt
def transcribeAudio(request):
    model = asr_model.Keyword_Spotting_Service()
    context = {}
    print(1)
    serverReceiveTime = time.time()
    audioData = request.FILES['data']
    sampleRate = int(request.POST['frameRate'])
    channelCount = int(request.POST['nChannels'])
    
    sampleWidth = int(request.POST['sampleWidth'])
    
    blob = audioData.read()
    # print(blob)
    webmFilePath = 'voice2text/audios/'+str(serverReceiveTime)+'.webm'
    with open(webmFilePath, 'wb') as f_aud:
        f_aud.write(blob)
    print("done")
    wavFilePath = webmFilePath[:-5] + '.wav'
    convert_webm_to_wav(webmFilePath, wavFilePath)
    print(model)
    output = model.predict(wavFilePath)
    # context['metadata'] = audioData.size
    # blob = audioData.read()
    # audio = wave.open('voice2text/audios/test3.wav', 'wb')
    # audio.setnchannels(2)
    # print(blob)
    # audio.setsampwidth(2)
    # audio.setframerate(44100)
    
    # audio.writeframes(blob) #on playing 'test.wav' only noise can be heard
    serverFinishTime = time.time()
    context['server receive time'] = serverReceiveTime
    context['server finish time'] = serverFinishTime
    context['output'] = output
    return JsonResponse(context)