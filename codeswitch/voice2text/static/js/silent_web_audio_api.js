


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

function combineText(messageList) {
  return messageList.join(" ");
}

// var chunkLenField = document.createElement("chunks-len");
// chunkLenField.setAttribute("type", "hidden");
// chunkLenField.value = 0;
// var messages = [];

var messageNum = 0;
// var audioBytesLen = 0;
var confirmButton;
var rejectButton;
var waitMessageBox;
var record;
var choiceButtons;
var recording = false;
var silentInstancesCnt = 0;
var nonSilentInstancesCnt = 0;
var indexed_transcript_chunks = [];
var furtherest_chunk_len = 0;

function cleanUpButtonsAndWaitMessage() {
  console.log("cleaning");
  document.getElementById("transcribe-text").removeChild(waitMessageBox);
  document.getElementById("transcribe-text").removeChild(choiceButtons);
  record.disabled = false;
}

function autoCorrect() {
  document.getElementById("transcribe-text-"+(messageNum-1)).innerHTML = document.getElementById("transcribe-text-"+(messageNum-1)+"-waitmessage").innerHTML;
  cleanUpButtonsAndWaitMessage();
}

async function getVolumes() {
  const stream1 = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
  const audioContext1 = new AudioContext();
  const mediaStreamAudioSourceNode1 = audioContext1.createMediaStreamSource(stream1);
  const analyserNode1 = audioContext1.createAnalyser();
  mediaStreamAudioSourceNode1.connect(analyserNode1);
  const pcmData1 = new Float32Array(analyserNode1.fftSize);
  setInterval(() => {
    analyserNode1.getFloatTimeDomainData(pcmData1);
    let sumSquares = 0.0;
    for (const amplitude of pcmData1) { sumSquares += amplitude*amplitude; }
    console.log(sumSquares / pcmData1.length);
    if (sumSquares / pcmData1.length < 10e-6) {
      silentInstancesCnt += 1;
    }
    else {
      nonSilentInstancesCnt += 1;
    }
  }, 100);
}

function combineText(messageList) {
  var resList = [];
  for (let i = 0; i < messageList.length; i++) {
    resList.push(messageList[i][0]);
  }
  return resList.join(" ");
}

// if (confirmButton) {
//   console.log("clicked confirm");
//   confirmButton.onclick = autoCorrect;
// }

// if (rejectButton) {
//   console.log("clicked ignore");
//   rejectButton.onclick = cleanUpButtonsAndWaitMessage;
// }

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

if (navigator.mediaDevices) {
  console.log('getUserMedia supported.');

  var constraints = { audio: true };
  var chunks = [];
  var lastBlob = null;
  var header = null;
  
  



  navigator.mediaDevices.getUserMedia(constraints)
  .then(function(stream) {
    var mediaRecorder = new MediaRecorder(stream, {mimeType : 'audio/webm', codecs : "pcm"});
    mediaRecorder.ondataavailable = function(e){
      chunks.push(e.data);
      // console.log("on data available got new chunks")
      // console.log(chunks);
      // printChunk(chunks);
      // console.log(e.data);
      processAudioChunk(false);
      // console.log("finished processing chunks, clearing chunks now")
      // console.log(chunks);
      // console.log(e.data);
      // console.log(header.type);
      // chunks = [header]; 
      // console.log("chunks cleared")
      // console.log(chunks);
      // console.log(e.data);
      // mediaRecorder.requestData();
    }
    

    record = document.getElementById("btn-toggle-audio");
    
    record.onclick = function() {
      if (recording) {
        mediaRecorder.stop();
        console.log(mediaRecorder.state);
        console.log("recorder stopped");
        record.style.background = "";
        record.style.color = "";
        waitMessageBox = document.createElement("div");
        waitMessageBox.id = "transcribe-text-"+messageNum+"-waitmessage";
        waitMessageBox.innerHTML = "Generating transcript using full audio ...";
        document.getElementById("transcribe-text").appendChild(waitMessageBox);
        messageNum += 1;
      }
      else {
        var messageBox = document.createElement("div");
        messageBox.id = "transcribe-text-"+messageNum;
        document.getElementById("transcribe-text").appendChild(messageBox);
        mediaRecorder.start(500);
        getVolumes();
        console.log(mediaRecorder.state);
        console.log("recorder started");
        record.style.background = "red";
        record.style.color = "black";
      }
      recording = !recording;
    }

    function processAudioChunk(runFull) {
      console.log("data available after MediaRecorder.stop() called.");      
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
      // volumeCallback();
      console.log("furtherest_chunk_len vs indexed_transcript_chunks.length", furtherest_chunk_len, indexed_transcript_chunks.length)
      if (runFull || (silentInstancesCnt >= 5 && furtherest_chunk_len <= indexed_transcript_chunks.length)) {
        silentInstancesCnt = 0;
        if (nonSilentInstancesCnt < 5 && !runFull) {
          return;
        }
        nonSilentInstancesCnt = 0;
        var data = new FormData();
        var audioStreamMeta = stream.getAudioTracks()[0].getSettings();
        if (indexed_transcript_chunks.length == 0) {
          data.append('lastlen', 0);
        }
        else {
          data.append('lastlen', indexed_transcript_chunks[indexed_transcript_chunks.length-1][2]);
        }
        furtherest_chunk_len += 1;
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
        data.append("messageNum", messageNum);
        data.append("runFull", runFull);
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
                console.log("transcribe-text-"+jsonResponse["messageNum"]);
                if (document.getElementById("transcribe-text-"+jsonResponse["messageNum"]) && recording) {
                  indexed_transcript_chunks.push([jsonResponse["output"], jsonResponse["server receive time"], jsonResponse["fullLen"]]);
                  indexed_transcript_chunks = indexed_transcript_chunks.sort((a, b) => a[1] - b[1]);
                  if (jsonResponse["messageNum"] == messageNum) {
                    document.getElementById("transcribe-text-"+jsonResponse["messageNum"]).innerHTML = combineText(indexed_transcript_chunks);
                  }
                  else {
                    document.getElementById("transcribe-text-"+jsonResponse["messageNum"]).innerHTML = jsonResponse["output"];
                  }
                }
                else if (runFull) {
                  document.getElementById("transcribe-text-"+(jsonResponse["messageNum"]-1)+"-waitmessage").innerHTML = jsonResponse["output"];
                  choiceButtons = document.createElement("div");
                  choiceButtons.id = "choice-buttons";
                  confirmButton = document.createElement("BUTTON");
                  confirmButton.innerText = "Replace";
                  rejectButton = document.createElement("BUTTON");
                  rejectButton.innerText = "Ignore";
                  record.disabled = true;
                  choiceButtons.appendChild(confirmButton);
                  choiceButtons.appendChild(rejectButton);
                  waitMessageBox.innerHTML = jsonResponse["output"];
                  document.getElementById("transcribe-text").appendChild(waitMessageBox);
                  document.getElementById("transcribe-text").appendChild(choiceButtons);
                  confirmButton.addEventListener("click", autoCorrect);
                  rejectButton.addEventListener("click", cleanUpButtonsAndWaitMessage);
                  indexed_transcript_chunks = [];
                  furtherest_chunk_len = 0;
                }
              }  
            }
        };
      }
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
      processAudioChunk(true);
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




// function printChunk(chunk) {
//   var arrayPromise = new Promise(function(resolve) {
//     var reader = new FileReader();

//     reader.onloadend = function() {
//         resolve(reader.result);
//     };

//     reader.readAsArrayBuffer(chunk);
//   });
//   var dBArray = [];

//   arrayPromise.then(function(array) {
//       console.log("Array contains", array, "bytes.");
//       var uint8View = new Uint8Array(array);
//       // console.log(uint8View);
//       for (let i = 0; i < uint8View.length-4; i++) {
//         dBArray.push(20*Math.log10(uint8View.slice(i,i+1)[0]));
//         if (i < 3) {
//           // console.log(parseInt(uint8View.slice(i, i+4)), uint8View.slice(i, i+4));
//           // console.log(uint8View.slice(i,i+1)[0], uint8View.slice(i+1,i+2)[0])
//         }
        
//         if (uint8View.slice(i,i+1)[0] == 31 && uint8View.slice(i+1,i+2)[0] == 67 && uint8View.slice(i+2,i+3)[0] == 182 && uint8View.slice(i+3,i+4)[0] == 117) {
//           console.log("found cluster head", i);
//         }
//       }
//       console.log(dBArray);
//   });
// }

// function combineText(messageList) {
//   return messageList.join(" ");
// }

// // var chunkLenField = document.createElement("chunks-len");
// // chunkLenField.setAttribute("type", "hidden");
// // chunkLenField.value = 0;
// // var messages = [];

// var messageNum = 0;
// // var audioBytesLen = 0;
// var confirmButton;
// var rejectButton;
// var waitMessageBox;
// var record;
// var choiceButtons;
// var silentChunkMessages = [];
// var chunkNumBackend = 0;
// let audioCtx = new AudioContext();
// let source;
// let analyzer = audioCtx.createAnalyser();
// analyzer.connect(audioCtx.destination);
// let javascriptNode;
// let amplitudeArray;

// function cleanUpButtonsAndWaitMessage() {
//   console.log("cleaning");
//   document.getElementById("transcribe-text").removeChild(waitMessageBox);
//   document.getElementById("transcribe-text").removeChild(choiceButtons);
//   record.disabled = false;
// }

// function autoCorrect() {
//   document.getElementById("transcribe-text-"+(messageNum-1)).innerHTML = document.getElementById("transcribe-text-"+(messageNum-1)+"-waitmessage").innerHTML;
//   cleanUpButtonsAndWaitMessage();
// }

// // if (confirmButton) {
// //   console.log("clicked confirm");
// //   confirmButton.onclick = autoCorrect;
// // }

// // if (rejectButton) {
// //   console.log("clicked ignore");
// //   rejectButton.onclick = cleanUpButtonsAndWaitMessage;
// // }

// function getCSRFToken() {
//   let cookies = document.cookie.split(";")
//   for (let i = 0; i < cookies.length; i++) {
//       let c = cookies[i].trim()
//       if (c.startsWith("csrftoken=")) {
//           return c.substring("csrftoken=".length, c.length)
//       }
//   }
//   return "unknown";
// }

// function printDB(data) {
//   for (let i = 0; i < data.length; i++) {
//     data[i] = 2*Math.log10(data[i]);
//   }
//   console.log("dB", data);
// }


// if (navigator.mediaDevices) {
//   console.log('getUserMedia supported.');

//   var constraints = { audio: true };
//   var chunks = [];
//   var recording = false;
//   var lastBlob = null;
//   var header = null;

//   navigator.mediaDevices.getUserMedia(constraints)
//   .then(function(stream) {
//     var mediaRecorder = new MediaRecorder(stream, {mimeType : 'audio/webm', codecs : "opus"});
//     audioCtx = new AudioContext();
//     source = audioCtx.createMediaStreamSource(stream);
//     mediaRecorder.ondataavailable = function(e){
//       chunks.push(e.data);
//       // printChunk(e.data);

//     // printChunk(chunks[chunks.length-1])
//       // console.log("on data available got new chunks")
//       // console.log(chunks);
//       // printChunk(chunks);
//       // console.log(e.data);
//       console.log(source);
//       processAudioChunk(false, false);
//       // console.log("finished processing chunks, clearing chunks now")
//       // console.log(chunks);
//       // console.log(e.data);
//       if (header == null) {
//         header = e.data.slice(0,150, type='audio/webm; codecs=opus');
//       }
//       // console.log(header.type);
//       // chunks = [header]; 
//       // console.log("chunks cleared")
//       // console.log(chunks);
//       // console.log(e.data);
//       // mediaRecorder.requestData();
//     }
    

//     record = document.getElementById("btn-toggle-audio");
    
//     record.onclick = function() {
//       if (recording) {
//         mediaRecorder.stop();
//         console.log(mediaRecorder.state);
//         console.log("recorder stopped");
//         record.style.background = "";
//         record.style.color = "";
//         waitMessageBox = document.createElement("div");
//         waitMessageBox.id = "transcribe-text-"+messageNum+"-waitmessage";
//         waitMessageBox.innerHTML = "autocorrecting ...";
//         document.getElementById("transcribe-text").appendChild(waitMessageBox);
//         messageNum += 1;
//       }
//       else {
//         var messageBox = document.createElement("div");
//         messageBox.id = "transcribe-text-"+messageNum;
//         document.getElementById("transcribe-text").appendChild(messageBox);
//         mediaRecorder.start(2000);
//         console.log(mediaRecorder.state);
//         console.log("recorder started");
//         record.style.background = "red";
//         record.style.color = "black";
//       }
//       recording = !recording;
//     }

//     // Watch the changes in the audio buffer
//     function viewBufferData() {
//       freqs = new Uint8Array(analyzer.frequencyBinCount);
//       times = new Uint8Array(analyzer.frequencyBinCount);
//       analyzer.smoothingTimeConstant = 0.8;
//       analyzer.fftSize = 2048;
//       analyzer.getByteFrequencyData(freqs);
//       analyzer.getByteTimeDomainData(times);
//       console.log(freqs)
//       console.log(times)
//     }

//     function processAudioChunk(lastChunk, runFull) {
//       console.log("data available after MediaRecorder.stop() called.");      
//       // var clipContainer = document.createElement('article');
//       // var clipLabel = document.createElement('p');
//       // var audio = document.createElement('audio');
//       // var deleteButton = document.createElement('button');

//       // clipContainer.classList.add('clip');
//       // audio.setAttribute('controls', '');
//       // deleteButton.innerHTML = "Delete";
//       // clipLabel.innerHTML = clipName;

//       // clipContainer.appendChild(audio);
//       // clipContainer.appendChild(clipLabel);
//       // clipContainer.appendChild(deleteButton);
//       // soundClips.appendChild(clipContainer);
//       // audio.controls = true;
//       audioCtx.decodeAudioData
//       console.log("stream", stream.getAudioTracks())
//       lastBlob = new Blob(chunks, { 'type' : 'audio/webm; codecs=opus'});
//       console.log(lastBlob);
//       var amplitudeArray = new Float32Array(analyzer.fftSize/2);
//       analyzer.getFloatFrequencyData(amplitudeArray);
//       console.log(amplitudeArray);
//       var data = new FormData();
//       var audioStreamMeta = stream.getAudioTracks()[0].getSettings();
//       if (lastBlob == null) {
//         data.append('lastlen', 0);
//       }
//       else {
//         data.append('lastlen', lastBlob.size);
//       }
      
//       // chunks = [];
    
//       // audio.src = audioURL;

//       // deleteButton.onclick = function(e) {
//       //   evtTgt = e.target;
//       //   evtTgt.parentNode.parentNode.removeChild(evtTgt.parentNode);
//       // }

//       var xhttp = new XMLHttpRequest();
//       xhttp.open("POST", "transcribeAudio", true);
      
//       data.append('data', lastBlob, 'audio_blob');
//       data.append('frameRate', audioStreamMeta.sampleRate);
//       data.append('sampleWidth', 2);
//       data.append('nChannels', audioStreamMeta.channelCount);
//       data.append("csrfmiddlewaretoken", getCSRFToken());
//       data.append("messageNum", messageNum);
//       data.append("silentChunkNum", chunkNumBackend);
//       data.append("lastChunk", lastChunk);
//       data.append("silentDetectionOn", true);
//       data.append("runFull", runFull);
//       xhttp.send(data);
      
//       xhttp.onreadystatechange = function() {
//           if (this.readyState == 4 && this.status == 200) {
//             // var audiometa = document.createElement('audiometa');
//             // audiometa.innerHTML = this.responseText;
//             // clipContainer.appendChild(audiometa);    
//             console.log(this.responseText);
//             var jsonResponse = JSON.parse(this.responseText);
//             if (jsonResponse["output"]) {
//               console.log("transcribe-text-"+jsonResponse["messageNum"]);
//               console.log(parseInt(jsonResponse["audioBytesLen"]), parseInt(jsonResponse["newSilentChunkNum"]), chunkNumBackend);
//               // if (jsonResponse["messageNum"] != messageNum) {
//               //   if (parseInt(jsonResponse["newSilentChunkNum"]) > chunkNumBackend && parseInt(jsonResponse["audioBytesLen"]) > 0) {
//               //     if (document.getElementById("transcribe-text-"+jsonResponse["messageNum"]) && recording) {
//               //       console.log(silentChunkMessages, jsonResponse["output"]);
//               //       silentChunkMessages.push(jsonResponse["output"]);
//               //       document.getElementById("transcribe-text-"+jsonResponse["messageNum"]).innerHTML = combineText(silentChunkMessages);
//               //       console.log("old chunknumbackend: ", chunkNumBackend, "new value: ", parseInt(jsonResponse["newSilentChunkNum"]));
//               //       chunkNumBackend = parseInt(jsonResponse["newSilentChunkNum"]);
  
//               //     }
//               //   document.getElementById("transcribe-text-"+jsonResponse["messageNum"]).innerHTML = document.getElementById("transcribe-text-"+jsonResponse["messageNum"]).innerHTML + jsonResponse["output"];
//               // }
//               if (runFull) {
//                 document.getElementById("transcribe-text-"+(jsonResponse["messageNum"]-1)+"-waitmessage").innerHTML = jsonResponse["output"];
//                 choiceButtons = document.createElement("div");
//                 choiceButtons.id = "choice-buttons";
//                 confirmButton = document.createElement("BUTTON");
//                 confirmButton.innerText = "Auto Correct";
//                 rejectButton = document.createElement("BUTTON");
//                 rejectButton.innerText = "Ignore";
//                 record.disabled = true;
//                 choiceButtons.appendChild(confirmButton);
//                 choiceButtons.appendChild(rejectButton);
//                 waitMessageBox.innerHTML = jsonResponse["output"];
//                 document.getElementById("transcribe-text").appendChild(waitMessageBox);
//                 document.getElementById("transcribe-text").appendChild(choiceButtons);
//                 confirmButton.addEventListener("click", autoCorrect);
//                 rejectButton.addEventListener("click", cleanUpButtonsAndWaitMessage);
//               }
//               else if (parseInt(jsonResponse["newSilentChunkNum"]) > chunkNumBackend && parseInt(jsonResponse["audioBytesLen"]) > 0) {
//                 if (document.getElementById("transcribe-text-"+jsonResponse["messageNum"]) && recording) {
//                   silentChunkMessages.push(jsonResponse["output"]);
//                   document.getElementById("transcribe-text-"+jsonResponse["messageNum"]).innerHTML = combineText(silentChunkMessages);
//                   console.log("old chunknumbackend: ", chunkNumBackend, "new value: ", parseInt(jsonResponse["newSilentChunkNum"]));
//                   chunkNumBackend = parseInt(jsonResponse["newSilentChunkNum"]);
//                 }
//               }
//             }
//             // if (jsonResponse["audioBytesLen"]) {
//             //   audioBytesLen = jsonResponse["audioBytesLen"];
//             // }
//             // var newChunk = false;
//             // if (jsonResponse["chunksNum"]) {
//             //   console.log(chunkLenField.value, jsonResponse["chunksNum"]);
//             //   newChunk = chunkLenField.value != jsonResponse["chunksNum"];
//             //   chunkLenField.value = jsonResponse["chunksNum"];
//             // }
//             // if (jsonResponse["output"]) {
//             //   if (newChunk) {
//             //     messages.push(jsonResponse["output"]);
//             //   }
//             //   else {
//             //     messages[-1] = jsonResponse["output"];
//             //   }
//             //   document.getElementById("transcribe-text").innerHTML = combineText(messages);
//             // }
            
            
//           }
//       };

//       // deleteButton.onclick = function(e) {
//       //   evtTgt = e.target;
//       //   evtTgt.parentNode.parentNode.removeChild(evtTgt.parentNode);
//       // }
//     }

//     // setInterval(()=>{
//     //   if (recording) {
//     //     processAudioChunk()
//     //   }
//     // }, 2000);
    

    

//     mediaRecorder.onstop = function(e) {
//       processAudioChunk(true, false);
//       processAudioChunk(false, true);
//       // chunks = [header];
//       chunks = [];
//       silentChunkMessages = [];
//     }

//     // mediaRecorder.ondataavailable = function(e) {
//     //   chunks.push(e.data);
//     // }
//   })
//   .catch(function(err) {
//     console.log('The following error occurred: ' + err);
//   })
// }