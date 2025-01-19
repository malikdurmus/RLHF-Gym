const socket = io();

let currentIndex = 0;
let feedback = [];
let videoPairs = [];

async function fetchVideoPairs() {
  try {
    const response = await fetch('/get_video_pairs');
    const data = await response.json();

    if (data.video_pairs && data.video_pairs.length > 0) {
      videoPairs = data.video_pairs;
      currentIndex = 0;
      displayVideoPair(videoPairs[currentIndex]);
      document.getElementById('status').innerText = "New video pairs loaded.";
    } else {
      document.getElementById('status').innerText = "No video pairs available. Waiting for new pairs to render..";
    }
  } catch (error) {
    console.error('Error fetching video pairs:', error);
    document.getElementById('status').innerText = "Failed to fetch video pairs.";
  }
}

function displayVideoPair(pair) {
  document.getElementById('video1').src = `/videos/${pair.video1}`;
  document.getElementById('video2').src = `/videos/${pair.video2}`;
  document.getElementById('status').innerText = `Pair ${currentIndex + 1} of ${videoPairs.length}`;
}
function setPreference(preference) {
  if (currentIndex >= videoPairs.length) return;

  const pair = videoPairs[currentIndex];
  feedback.push({ id: pair.id, preference });

  currentIndex++;
  if (currentIndex < videoPairs.length) {
    displayVideoPair(videoPairs[currentIndex]);
  } else {
    document.getElementById('status').innerText = "All pairs completed. Submitting feedback...";
    submitFeedback();
  }
}

async function submitFeedback() {
  try {
    await fetch('/submit_preferences', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ preferences: feedback })
    });

    document.getElementById('status').innerText = "Feedback submitted. Now waiting for new video pairs...";
    feedback = [];
    videoPairs = [];
    currentIndex = 0;
  } catch (error) {
    console.error('Error submitting feedback:', error);
    document.getElementById('status').innerText = "Failed to submit feedback.";
  }
}

socket.on('new_video_pairs', (data) => {
  console.log('New video pairs notification received:', data);
  fetchVideoPairs();
});

window.onload = function () {
  fetchVideoPairs();
};
// this runs initially fiest time
// TODO: need to ensure reloading doesnt raise errors in the train.py (it rose the following error once,
//  but couldnt reproduce the error again) "queue has more/less entries"
// TODO: submitted preference must be processed in the backend for rew net update
