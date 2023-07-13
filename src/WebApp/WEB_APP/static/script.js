document.addEventListener('DOMContentLoaded', function () {
    var userInputText = document.getElementById('userInputText');
    userInputText.addEventListener('input', updateCounts);

    var form = document.querySelector('.userInputInterface');
    // form.addEventListener('submit', showLoading);
    form.addEventListener('submit', function (event) {
        if (!validateForm()) {
            event.preventDefault(); 
        } else {
            showLoading()
        }
    });
});

function updateCounts() {
    var userInput = document.getElementById('userInputText').value;
    var wordCount = document.querySelector('.wordCount span');
    var charCount = document.querySelector('.charCount span');

    var words = userInput.trim().split(/\s+/);
    var numWords = words.length;
    var numChars = userInput.length;

    // Adjust word count if input is empty
    if (userInput.trim() === '') {
        numWords = 0;
    }

    wordCount.textContent = 'Word Count: ' + numWords;
    charCount.textContent = 'Character Count: ' + numChars;
}

function showLoading() {
    var loadingContainer = document.getElementById('loadingContainer');
    loadingContainer.style.display = 'block';
    uiTextbox.style.display = 'none';
    limitations.style.display = 'none';

    style="display: none;"

    // Simulate a delay of 15 seconds
    setTimeout(function () {
        var loadingContainer = document.getElementById('loadingContainer');
        loadingContainer.style.display = 'none';
    }, 15000);
}

function validateForm() {
    var userInputText = document.getElementById('userInputText');
    var words = userInputText.value.trim().split(/\s+/);
    if ((userInputText.value.trim() === '') || (words.length < 2)) {
        return false;
    }
    return true;
}
