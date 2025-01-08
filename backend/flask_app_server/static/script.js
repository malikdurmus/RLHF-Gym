const socket = io("http://localhost:5000"); // Replace with your backend URL if needed

socket.on("connect", () => {
  console.log("Connected to the server via WebSocket.");
});

socket.on("disconnect", () => {
  console.log("Disconnected from the server.");
});

// Listen for the 'videos_ready' event
// socket.on("videos_ready", (data) => {
//   console.log("Videos ready:", data.ready_videos);
//   if (data.ready_videos.length >= 2) {
//     // Assuming data.ready_videos contains paths for two videos
//     loadVideo(data.ready_videos[0], data.ready_videos[1]);
//   }
// });
socket.on("videos_ready", (data) => {
  console.log("Videos ready:", data);
});

function loadVideo(videoSrc1, videoSrc2) {
        const mainOption1 = document.getElementById('mainOption1');
        const mainOption2 = document.getElementById('mainOption2');
        const neutralOption = document.getElementById('neutralOption');
        const videoPlayer1 = document.getElementById('videoPlayer1');
        const videoPlayer2 = document.getElementById('videoPlayer2');
        const videoSource1 = document.getElementById('videoSource1');
        const videoSource2 = document.getElementById('videoSource2');
        const loader1 = document.getElementById('loader1');
        const loader2= document.getElementById('loader2');

    mainOption1.onclick = mainOption2.onclick = neutralOption.onclick = clicks;

    function clicks() {
        videoPlayer1.pause();
        videoPlayer2.pause();
        videoPlayer1.style.display = 'none';
        videoPlayer2.style.display = 'none';
        neutralOption.style.display = 'none';
        loader1.style.display = 'block';
        loader2.style.display = 'block';

        setTimeout(() => {
            videoSource1.src = videoSrc1;
            videoSource2.src = videoSrc2;
            videoPlayer1.load();
            videoPlayer2.load();
            loader1.style.display = 'none';
            loader2.style.display = 'none';
            videoPlayer1.style.display = 'block';
            videoPlayer2.style.display = 'block';
            neutralOption.style.display = 'block';
            videoPlayer1.play();
            videoPlayer2.play();
        }, 5000);
    }
}

