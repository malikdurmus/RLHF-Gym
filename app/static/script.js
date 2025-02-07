const socket = io(); // Socket connection to the server
let currentIndex = 0; // Current video pair index
let feedback = []; // Stores the user feedback
let videoPairs = []; // Stores the video pairs
let status;  // Stores current status message
let runName = ""; // Stores current run name

// Show the explanation modal and fetch the run name and video pairs once the window is loaded
window.onload = async function () {
  status = document.getElementById('status');
  displayExplanationModal();
  await fetchRunName();
  fetchVideoPairs();
};

// Update the status displayed on the page
function updateStatus(message) {
  if (status) {
    status.innerText = message;
  } else {
    console.warn('Status element is not defined.');
  }
}

// Fetch video pairs once new video pairs have been received
socket.on('new_video_pairs', (data) => {
  console.log('New video pairs notification received:', data);
  fetchVideoPairs();
});

// Fetch the run name from the server
async function fetchRunName() {
  try {
    const response = await fetch('/get_run_name'); // Get request to the corresponding endpoint
    const data = await response.json(); // Parse the JSON response
    runName = data.run_name;
    console.log("Run name fetched:", runName);
  } catch (error) {
    console.error('Error fetching run_name:', error);
  }
}

//Fetch the video pairs from the server
async function fetchVideoPairs() {
  try {
    const response = await fetch('/get_video_pairs'); // Get request to the corresponding endpoint
    const data = await response.json(); // Parse the JSON response

    // Check if the JSON response contains a video pair and then update global videoPairs variable
    if (data.video_pairs && data.video_pairs.length > 0) {
      videoPairs = data.video_pairs;
      currentIndex = 0;
      updateStatus("Info: New video pairs have been loaded. Please choose Pair 1");
      displayVideoPair(videoPairs[currentIndex]);
    } else {
      updateStatus("Info: No video pairs available at the moment. Waiting for new pairs...");
    }
  } catch (error) {
    console.error('Error fetching video pairs:', error);
    updateStatus("Info: Failed to fetch video pairs.");
  }
}

// Display the video pair and set sources to the video elements of the current pair
function displayVideoPair(pair) {
  document.getElementById('video1').src = `/videos/${runName}/${pair.video1}`;
  document.getElementById('video2').src = `/videos/${runName}/${pair.video2}`;
  updateStatus(`Pair ${currentIndex + 1} of ${videoPairs.length}`); // Show the current index and total number of pairs
}

// Set user's preference for the current video pair
function setPreference(preference) {
  if (currentIndex >= videoPairs.length) return; // Check if all video pairs have been processed

  const pair = videoPairs[currentIndex]; // Get the current video pair based on the index
  feedback.push({ id: pair.id, preference });  // Store video pair ID and the user's preference

  currentIndex++; // Move to the next video pair

  // Display the next video pair if there are more pairs
  if (currentIndex < videoPairs.length) {
    displayVideoPair(videoPairs[currentIndex]);
  } else {
    updateStatus("Info: All pairs completed. Submitting feedback...");
    submitFeedback();
  }
}

// Submit user feedback to the server
async function submitFeedback() {
  try {
    await fetch('/submit_preferences', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json' // Setting the content type to JSON
      },
      body: JSON.stringify({ preferences: feedback })  // Sending the feedback in the request body
    });
    updateStatus("Info: Feedback submitted. Now waiting for new video pairs...");

    // Resetting the feedback, video pairs and index after submission
    feedback = [];
    videoPairs = [];
    currentIndex = 0;
  } catch (error) {
    console.error('Error submitting feedback:', error);
    updateStatus("Info: Failed to submit feedback.");
  }
}

// Display the explanation modal and hide  main content 
function displayExplanationModal() {
    const modal = document.getElementById("explanationModal");
    const closeRules = document.getElementById("modalCloseButton");
    const mainContent = document.getElementById('mainContent');

    mainContent.classList.add('mainContent');
    modal.style.display = "flex";
    document.body.style.overflow = 'hidden'; // Disable scrolling

    // Close the modal and display the main content once close button is clicked
    closeRules.addEventListener("click", () => {
        modal.style.display = "none";
        mainContent.classList.remove('mainContent');
        document.body.style.overflow = 'auto'; // Enable scrolling
    });
}