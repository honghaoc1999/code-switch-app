from django.shortcuts import render
import wave
import json
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import base64
import subprocess
import time
# Create your views here.
def recorderView(request):
    context = {}
    return render(request, 'recorder.html', context)

def convert_webm_to_wav(file):
    command = ['ffmpeg', '-i', file, '-acodec', 'pcm_s16le', '-ac', '1', '-ar', '16000', file[:-5] + '.wav']
    subprocess.run(command,stdout=subprocess.PIPE,stdin=subprocess.PIPE)


@csrf_exempt
def transcribeAudio(request):
    context = {}
    print(1)
    print(request)
    print(request.FILES['data'])
    serverReceiveTime = time.time()
    audioData = request.FILES['data']
    sampleRate = int(request.POST['frameRate'])
    channelCount = int(request.POST['nChannels'])
    
    sampleWidth = int(request.POST['sampleWidth'])
    print("sampleWidth", sampleWidth, "channelCount", channelCount, "framerate", sampleRate)
    print(type(audioData)) 
    print(audioData.size)
    blob = audioData.read()
    print(blob)
    with open('voice2text/audios/'+str(serverReceiveTime)+'.webm', 'wb') as f_aud:
        f_aud.write(blob)
    print("done")
    convert_webm_to_wav('voice2text/audios/'+str(serverReceiveTime)+'.webm')
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
    return JsonResponse(context)