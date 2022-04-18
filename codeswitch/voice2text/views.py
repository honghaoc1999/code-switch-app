from django.shortcuts import render
import wave
import json
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import base64
import subprocess
import time
import voice2text.asr_model as asr_model
import os
import uuid
import logging
import re
# LOG_FILENAME = 'voice2text/logging_example.out'
# logging.basicConfig(filename=LOG_FILENAME, level=logging.DEBUG)
print("reached view")
# Create your views here.
def recorderView(request):
    context = {}
    model = asr_model.Keyword_Spotting_Service()
    return render(request, 'index.html', context)

def periodicRetryRecorder(request):
    context = {}
    model = asr_model.Keyword_Spotting_Service()
    return render(request, '3s_retry_recorder.html', context)

def periodic6sRetryRecorder(request):
    context = {}
    model = asr_model.Keyword_Spotting_Service()
    return render(request, '6s_retry_recorder.html', context)

def entireRetryRecorder(request):
    context = {}
    model = asr_model.Keyword_Spotting_Service()
    return render(request, 'entire_retry.html', context)

def silentChunkRecorder(request):
    context = {}
    model = asr_model.Keyword_Spotting_Service()
    return render(request, 'silent_chunking.html', context)

def convert_webm_to_wav(webmFile, wavFile):
    command = ['ffmpeg', '-fflags', '+igndts', '-i', webmFile,  '-acodec', 'pcm_s16le', '-ac', '1', '-ar', '16000', wavFile]
    subprocess.run(command,stdout=subprocess.PIPE,stdin=subprocess.PIPE)


# @csrf_exempt
def transcribeAudio(request):
    model = asr_model.Keyword_Spotting_Service()
    context = {}
    serverReceiveTime = time.time()
    audioData = request.FILES['data']
    sampleRate = int(request.POST['frameRate'])
    channelCount = int(request.POST['nChannels'])
    lastBlobStamp = int(request.POST['lastlen'])
    sampleWidth = int(request.POST['sampleWidth'])
    runFull = request.POST['runFull']
    if 'silentChunkNum' in request.POST:
        silentChunkNum = int(request.POST['silentChunkNum'])
    if 'lastChunk' in request.POST:
        lastChunk = request.POST['lastChunk']
    # messageLevel1Index = request.POST['messageLevel1Index']
    # overWriteLevel1Message = request.POST['overWriteLevel1Message']
    # lastAudioLen = int(request.POST['audioBytesLen'])
    
    blob = audioData.read()
    print("curr len ", len(list(blob)), "last blob", lastBlobStamp, "runFull", runFull)
    webmFilePath = 'voice2text/audios/'+str(uuid.uuid1())+'.webm'
    with open(webmFilePath, 'wb') as f_aud:
        f_aud.write(blob)
    print("done")
    wavFilePath = webmFilePath[:-5] + '.wav'
    try:
        convert_webm_to_wav(webmFilePath, wavFilePath)

    except:
        logging.exception('convert webm to wav failed')
    try:    
        if 'silentDetectionOn' in request.POST:
            output, audioBytesLen, newSilentChunkNum = model.predict(wavFilePath, lastBlobStamp, runFull, lastChunk, silentChunkNum=silentChunkNum)
        elif 'runEng' in request.POST:
            output, audioBytesLen, newSilentChunkNum = model.predict(wavFilePath, lastBlobStamp, runFull, runEng = request.POST['runEng'])
        else:
            output, audioBytesLen, newSilentChunkNum = model.predict(wavFilePath, lastBlobStamp, runFull)
        # print("last blob", lastBlobStamp, "curr blob", len(list(blob)), "wav len", audioBytesLen)
        serverFinishTime = time.time()
        context['server receive time'] = serverReceiveTime
        context['server finish time'] = serverFinishTime
        context['output'] = output
        context['audioBytesLen'] = audioBytesLen
        context['messageNum'] = int(request.POST['messageNum'])
        if newSilentChunkNum == None:
            context['newSilentChunkNum'] = -1
        else:
            context['newSilentChunkNum'] = newSilentChunkNum
        os.remove(wavFilePath)
        os.remove(webmFilePath)
        # context['messageLevel1Index'] = messageLevel1Index
        # context['overWriteLevel1Message'] = overWriteLevel1Message
        
    except:
        os.remove(wavFilePath)
        os.remove(webmFilePath)
        logging.exception('predict failed')
    return JsonResponse(context)

