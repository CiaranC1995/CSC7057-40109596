document.addEventListener('DOMContentLoaded', function () {
    var userInputText = document.getElementById('userInputText');
    userInputText.addEventListener('input', updateCounts);
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