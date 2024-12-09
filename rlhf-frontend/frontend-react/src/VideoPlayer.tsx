import { useState, useEffect, useRef} from 'react';

function VideoPlayer() {
    const [videoUrl, setVideoUrl] = useState('');
    const videoRef = useRef<HTMLVideoElement>(null);

    useEffect(() => {
        setVideoUrl('http://localhost:5000/video');
        //Right now the server returns http 206 meaning that it is loaded partially (in chunks)
        //but that is not a problem since the real video that will be produced by the gym environment
        // is smaller than the data buffer
    }, []);


    const resetVideo = () => {
        if (videoRef.current) {
            videoRef.current.currentTime = 0;
            videoRef.current.pause();
        }
    }

    return (
        <div>
            {videoUrl ? (
                <div>
                    <video ref= {videoRef} width="360" height="360" controls>
                        <source src={videoUrl} type="video/mp4" />
                    </video>
                        <div>
                            <button onClick={resetVideo}>Reset Video</button>
                        </div>
                    </div>
                ) : (
                    <p>Loading video...</p>
                )}
        </div>
    );
}

export default VideoPlayer;


