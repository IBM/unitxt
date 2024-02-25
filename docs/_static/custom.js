let element = document.createElement('script');
element.type = 'module';
element.src = 'https://gradio.s3-us-west-2.amazonaws.com/4.14.0/gradio.js';
document.head.appendChild(element);

element = document.createElement('link');
element.rel = 'stylesheet';
element.type = 'text/css';
element.href = "https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/tokyo-night.min.css";
document.head.appendChild(element);

function loadScript(src, callback) {
    var script = document.createElement('script');
    script.src = src;
    script.onload = function() {
        if (callback) callback();
    };
    script.onerror = function() {
        console.error('Failed to load script: ' + src);
    };
    document.head.appendChild(script);
}

loadScript('https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js', function() {
    loadScript('https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/languages/python.min.js', function() {
            hljs.highlightAll();
    })
})



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
