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

    mainOption1.onclick = mainOption2.onclick = neutralOption.onclick = clickOption;

    function clickOption() {
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

    mainOption1.onclick = () => sendFeedback(1);
    mainOption2.onclick = () => sendFeedback(2);
    neutralOption.onclick = () => sendFeedback(0);

    function sendFeedback(option) {

        const feedbackData = {
        preference: option === 1 ? "option1" : option === 2 ? "option2" : "neutral_option",
        neutralOption: option === 0 ? "neutral_option" : ""
   };

    fetch('/train-agent', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(feedbackData)
    })

    .then(response => response.json())
    .then(result => {
        console.log('Feedback received:', result);
    })
    .catch(error => {
        console.error('Error:', error);
    });
}
}