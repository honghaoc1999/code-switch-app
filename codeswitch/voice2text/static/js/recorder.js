if (navigator.mediaDevices) {
  console.log('getUserMedia supported.');

  var constraints = { audio: true };
  var chunks = [];
  var recording = false;

  navigator.mediaDevices.getUserMedia(constraints)
  .then(function(stream) {
    var mediaRecorder = new MediaRecorder(stream, {mimeType : 'audio/webm', codecs : "opus"});
    mediaRecorder.ondataavailable = function(e){
      chunks.push(e.data);
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
        mediaRecorder.start();
        console.log(mediaRecorder.state);
        console.log("recorder started");
        record.style.background = "red";
        record.style.color = "black";
      }
      recording = !recording;
    }

    setInterval(()=>{
      if (recording) {
        processAudioChunk()
      }
    }, 2000);

    function processAudioChunk() {
      console.log("data available after MediaRecorder.stop() called.");

      var clipName = Date.now();

      var clipContainer = document.createElement('article');
      var clipLabel = document.createElement('p');
      var audio = document.createElement('audio');
      var deleteButton = document.createElement('button');

      clipContainer.classList.add('clip');
      audio.setAttribute('controls', '');
      deleteButton.innerHTML = "Delete";
      clipLabel.innerHTML = clipName;

      clipContainer.appendChild(audio);
      clipContainer.appendChild(clipLabel);
      clipContainer.appendChild(deleteButton);
      soundClips.appendChild(clipContainer);
      audio.controls = true;
      
      var audioStreamMeta = stream.getAudioTracks()[0].getSettings();
      console.log(audioStreamMeta);
      console.log(chunks);
      var blob = new Blob(chunks, { 'type' : 'audio/webm; codecs=opus'});
      console.log(blob);
      chunks = [];
      var audioURL = URL.createObjectURL(blob);
      
      audio.src = audioURL;

      deleteButton.onclick = function(e) {
        evtTgt = e.target;
        evtTgt.parentNode.parentNode.removeChild(evtTgt.parentNode);
      }

      var xhttp = new XMLHttpRequest();
      xhttp.open("POST", "transcribeAudio", true);
      var data = new FormData();
      data.append('data', blob, 'audio_blob');
      data.append('frameRate', audioStreamMeta.sampleRate);
      data.append('sampleWidth', 2);
      data.append('nChannels', audioStreamMeta.channelCount);
      console.log(audioStreamMeta.frameRate);
      console.log(audioStreamMeta.width);
      xhttp.send(data);
      xhttp.onreadystatechange = function() {
          if (this.readyState == 4 && this.status == 200) {
            var audiometa = document.createElement('audiometa');
            audiometa.innerHTML = this.responseText;
            clipContainer.appendChild(audiometa);    
          }
      };

      deleteButton.onclick = function(e) {
        evtTgt = e.target;
        evtTgt.parentNode.parentNode.removeChild(evtTgt.parentNode);
      }
    }

    

    

    mediaRecorder.onstop = function(e) {
      processAudioChunk();
    }

    // mediaRecorder.ondataavailable = function(e) {
    //   chunks.push(e.data);
    // }
  })
  .catch(function(err) {
    console.log('The following error occurred: ' + err);
  })
}
