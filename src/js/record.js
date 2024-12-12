async function startRecording() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        const mediaRecorder = new MediaRecorder(stream);
        const chunks = [];

        mediaRecorder.ondataavailable = (e) => chunks.push(e.data);
        mediaRecorder.onstop = () => {
            const blob = new Blob(chunks, { type: 'video/webm' });
            const url = URL.createObjectURL(blob);
            const video = document.createElement('video');
            video.src = url;
            video.controls = true;
            document.body.appendChild(video);
        };

        mediaRecorder.start();
        setTimeout(() => mediaRecorder.stop(), 5000); // Stop after 5 seconds
    } catch (err) {
        console.error('Error recording video:', err);
    }
}