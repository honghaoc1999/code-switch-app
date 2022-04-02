"use strict"
console.log("reached here");
function printChunk(chunk) {
  var arrayPromise = new Promise(function(resolve) {
    var reader = new FileReader();

    reader.onloadend = function() {
        resolve(reader.result);
    };

    reader.readAsArrayBuffer(chunk[0]);
  });

  arrayPromise.then(function(array) {
      console.log("Array contains", array, "bytes.");
      var uint8View = new Uint8Array(array);
      console.log(uint8View);
      for (let i = 0; i < uint8View.length-4; i++) {
        if (i < 3) {
          console.log(parseInt(uint8View.slice(i, i+4)), uint8View.slice(i, i+4));
          console.log(uint8View.slice(i,i+1)[0], uint8View.slice(i+1,i+2)[0])
        }
        
        if (uint8View.slice(i,i+1)[0] == 31 && uint8View.slice(i+1,i+2)[0] == 67 && uint8View.slice(i+2,i+3)[0] == 182 && uint8View.slice(i+3,i+4)[0] == 117) {
          console.log("found cluster head", i);
        }
      }
  });
}

function getCSRFToken() {
  let cookies = document.cookie.split(";")
  for (let i = 0; i < cookies.length; i++) {
      let c = cookies[i].trim()
      if (c.startsWith("csrftoken=")) {
          return c.substring("csrftoken=".length, c.length)
      }
  }
  return "unknown";
}
console.log(navigator.mediaDevices);
if (navigator.mediaDevices) {
  console.log('getUserMedia supported.');

  var constraints = { audio: true };
  var chunks = [];
  var recording = false;
  var lastBlob = null;
  var header = null;

  navigator.mediaDevices.getUserMedia(constraints)
  .then(function(stream) {
    var mediaRecorder = new MediaRecorder(stream, {mimeType : 'audio/webm', codecs : "opus"});
    mediaRecorder.ondataavailable = function(e){
      chunks.push(e.data);
      // console.log("on data available got new chunks")
      // console.log(chunks);
      // printChunk(chunks);
      // console.log(e.data);
      processAudioChunk();
      // console.log("finished processing chunks, clearing chunks now")
      // console.log(chunks);
      // console.log(e.data);
      if (header == null) {
        header = e.data.slice(0,150, type='audio/webm; codecs=opus');
      }
      // console.log(header.type);
      // chunks = [header]; 
      // console.log("chunks cleared")
      // console.log(chunks);
      // console.log(e.data);
      // mediaRecorder.requestData();
    }
    

    var record = document.getElementById("btn-toggle-audio");
    var soundClips = document.getElementById("audio-clips");
    
    record.onclick = function() {
      if (recording) {
        mediaRecorder.stop();
        console.log(mediaRecorder.state);
        console.log("recorder stopped");
        record.style.background = "";
        record.style.color = "";
      }
      else {
        mediaRecorder.start(1000);
        console.log(mediaRecorder.state);
        console.log("recorder started");
        record.style.background = "red";
        record.style.color = "black";
      }
      recording = !recording;
    }

    function processAudioChunk() {
      console.log("data available after MediaRecorder.stop() called.");
      var clipName = Date.now();
      
      // var clipContainer = document.createElement('article');
      // var clipLabel = document.createElement('p');
      // var audio = document.createElement('audio');
      // var deleteButton = document.createElement('button');

      // clipContainer.classList.add('clip');
      // audio.setAttribute('controls', '');
      // deleteButton.innerHTML = "Delete";
      // clipLabel.innerHTML = clipName;

      // clipContainer.appendChild(audio);
      // clipContainer.appendChild(clipLabel);
      // clipContainer.appendChild(deleteButton);
      // soundClips.appendChild(clipContainer);
      // audio.controls = true;
      var data = new FormData();
      var audioStreamMeta = stream.getAudioTracks()[0].getSettings();
      if (lastBlob == null) {
        data.append('lastlen', 0);
      }
      else {
        data.append('lastlen', lastBlob.size);
      }
      console.log(chunks);
      lastBlob = new Blob(chunks, { 'type' : 'audio/webm; codecs=opus'});
      console.log(lastBlob);
      // chunks = [];
    
      // audio.src = audioURL;

      // deleteButton.onclick = function(e) {
      //   evtTgt = e.target;
      //   evtTgt.parentNode.parentNode.removeChild(evtTgt.parentNode);
      // }

      var xhttp = new XMLHttpRequest();
      xhttp.open("POST", "transcribeAudio", true);
      
      data.append('data', lastBlob, 'audio_blob');
      data.append('frameRate', audioStreamMeta.sampleRate);
      data.append('sampleWidth', 2);
      data.append('nChannels', audioStreamMeta.channelCount);
      data.append("csrfmiddlewaretoken", getCSRFToken());
      console.log(audioStreamMeta.frameRate);
      console.log(audioStreamMeta.width);
      console.log(audioStreamMeta);
      xhttp.send(data);
      
      xhttp.onreadystatechange = function() {
          if (this.readyState == 4 && this.status == 200) {
            // var audiometa = document.createElement('audiometa');
            // audiometa.innerHTML = this.responseText;
            // clipContainer.appendChild(audiometa);    
            console.log(this.responseText);
            var jsonResponse = JSON.parse(this.responseText);
            if (jsonResponse["output"]) {
              document.getElementById("transcribe-text").innerHTML = document.getElementById("transcribe-text").innerHTML + ' ' + jsonResponse["output"];
            }
            
          }
      };

      // deleteButton.onclick = function(e) {
      //   evtTgt = e.target;
      //   evtTgt.parentNode.parentNode.removeChild(evtTgt.parentNode);
      // }
    }

    // setInterval(()=>{
    //   if (recording) {
    //     processAudioChunk()
    //   }
    // }, 2000);
    

    

    mediaRecorder.onstop = function(e) {
      processAudioChunk();
      // chunks = [header];
      chunks = [];
    }

    // mediaRecorder.ondataavailable = function(e) {
    //   chunks.push(e.data);
    // }
  })
  .catch(function(err) {
    console.log('The following error occurred: ' + err);
  })
}
