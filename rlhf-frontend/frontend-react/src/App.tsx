import { useEffect, useState } from 'react';
import './App.css';
import VideoPlayer from "./VideoPlayer.tsx";


function App() {

    interface ApiResponse {
    members: [];
}
    const [data, setData] = useState([]);

    useEffect(() => {
        // Fetch members data after the component mounts
        fetch("http://127.0.0.1:5000/members")
            .then(res => {
                if (!res.ok) {
                    throw new Error("response not ok");
                }
                return res.json();
            })
            .then((data: ApiResponse) => {
                setData(data.members);
                console.log(data.members);
            })
            .catch(err => {
                console.error("failed to fetch:", err);
            });
    }, []); //update the empty dependency array

    const handleClick = () => (
        //send preference
        //get the new video
        console.log(2)
    )

    return (
        <div>
            <h1>Members</h1>
            <ul>
                {data.map((member, index) => (
                    <li key={index}>{member}</li> // Display member's name
                ))}
            </ul>
            <VideoPlayer/>
            <VideoPlayer/>
            <button onClick={handleClick}> Button1</button>
            <button onClick={handleClick}> Button2</button>
        </div>
    );
}

export default App;
