let element = document.createElement('script');
element.type = 'module';
element.src = 'https://gradio.s3-us-west-2.amazonaws.com/4.14.0/gradio.js';
element.onload = function() {
    console.log('Successfully loaded: ' + element.src);
    function handleLoadComplete() {
        console.log("Embedded space has finished rendering");
        const loadingElement = document.querySelector(".gradio-loading");
        if (loadingElement) {
            loadingElement.style.display = 'none';
        }
    }

    const gradioApp = document.querySelector("gradio-app");
    gradioApp.addEventListener("render", handleLoadComplete);
};
element.onerror = function() {
    console.error('Failed to load script: ' + element.src);
};
document.head.appendChild(element);


document.addEventListener('DOMContentLoaded', function() {
    var videoElement = document.getElementById('controlled-video');

    // Check if the video element exists
    if (videoElement) {
        // Mouseover event to add controls
        videoElement.addEventListener('mouseover', function() {
            this.setAttribute('controls', 'controls');
        });

        // Mouseout event to remove controls
        videoElement.addEventListener('mouseout', function() {
            this.removeAttribute('controls');
        });
    }
});

document.addEventListener("DOMContentLoaded", () => {
    const switches = document.querySelectorAll(".switch input[type='checkbox']"); // Select all switches
    switches.forEach(switchElement => {
      switchElement.addEventListener("change", function() {
        const isChecked = this.checked;
        // Determine which type of code snippets to show based on the checkbox state
        const typeToShow = isChecked ? "noInstallation" : "withInstallation";

        // Update all switches to reflect this switch's state
        switches.forEach(switchEl => {
          switchEl.checked = isChecked; // Synchronize switch states
        });

        // Update code snippets visibility
        setActiveCodeSnippets(typeToShow);
      });
    });
});

function setActiveCodeSnippets(type) {
    const allSnippets = document.querySelectorAll(".code-snippet");
    allSnippets.forEach(snippet => {
      snippet.style.display = snippet.getAttribute("data-code") === type ? 'block' : 'none';
    });
}


