import { useState, useEffect } from 'react';

function VideoPlayer() {
    const [videoUrl, setVideoUrl] = useState('');

    useEffect(() => {
        setVideoUrl('http://localhost:5000/video');
        //Right now the server returns http 206 meaning that it is loaded partially (in chunks)
        //but that is not a problem since the real video that will be produced by the gym environment
        // is smaller than the data buffer
    }, []);

    return (
        <div>
            {videoUrl ? (
                <video width="360" height="360" controls>
                    <source src={videoUrl} type="video/mp4" />
                </video>
            ) : (
                <p>Loading video...</p>
            )}
        </div>
    );
}

export default VideoPlayer;