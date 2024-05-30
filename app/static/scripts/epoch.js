// epoch.js

$(document).ready(function() {
    
    $('#create-model-button').on('click', function() {
        var totalTimeSteps = parseInt($('#t_steps').val());
        var forecastSteps = parseInt($('#fcst_steps').val());

        if (!totalTimeSteps || !forecastSteps) {
            alert('Please enter valid time steps and forecast steps!');
            return;
        }

        var $progressBar = $('.progress');
        var $progressText = $('.progress');
        var currentProgress = 0;
        var totalEpochs = parseInt($('#epoch_val').val()); 

        var interval = setInterval(function() {
            currentProgress++;

            var progressPercentage = (currentProgress / totalEpochs * 100).toFixed(0);
            $progressBar.css('width', progressPercentage + '%');
            $progressText.text(progressPercentage + '%');

            if (currentProgress >= totalEpochs) {
                clearInterval(interval);

                $('#lstm-form').submit();
            }
        }, 1000); // Update every second
    });
});
