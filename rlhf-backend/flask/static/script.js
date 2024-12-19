function loadVideo(videoSrc) {
    const videoPlayer = document.getElementById('videoPlayer');
    const videoSource = document.getElementById('videoSource');
    videoSource.src = videoSrc;
    videoPlayer.load();
    videoPlayer.play();
}