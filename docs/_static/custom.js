let script = document.createElement('script');
script.type = 'module';
script.src = 'https://gradio.s3-us-west-2.amazonaws.com/4.14.0/gradio.js';
document.head.appendChild(script);

document.addEventListener('DOMContentLoaded', function() {
    var videoElement = document.getElementById('controlled-video');

    // Mouseover event to add controls
    videoElement.addEventListener('mouseover', function() {
        this.setAttribute('controls', 'controls');
    });

    // Mouseout event to remove controls
    videoElement.addEventListener('mouseout', function() {
        this.removeAttribute('controls');
    });
});
